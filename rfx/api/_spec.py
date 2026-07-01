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
    # RO4003C: this entry uses the *process* Dk 3.55; for 50-ohm impedance
    # synthesis most designs use the *design* Dk 3.38 (Rogers RO4000 datasheet).
    "rogers4003c": {"eps_r": 3.55, "sigma": 0.0027 * 2 * np.pi * 5e9 * 3.55 * EPS_0},
    # RO4350B: Dk(design) 3.48, Df 0.0037 @ 10 GHz (Rogers RO4000 datasheet).
    "rogers4350b": {"eps_r": 3.48, "sigma": 0.0037 * 2 * np.pi * 10e9 * 3.48 * EPS_0},
    # RT/duroid 5880: Dk 2.20, Df 0.0009 @ 10 GHz (Rogers RT/duroid datasheet).
    "rt_duroid_5880": {"eps_r": 2.20, "sigma": 0.0009 * 2 * np.pi * 10e9 * 2.20 * EPS_0},
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
    ``ad_segmented_gb`` for the runner-supported segmented paths:
    ``checkpoint_every`` on the non-uniform scan-of-scan path and
    ``checkpoint_segments`` on the uniform segmented-scan path. When present,
    ``ad_segmented_active_segments`` is the actual number of active segment
    boundaries/carry snapshots used by the segmented estimate after warmup.
    ``evidence_class`` labels this artifact as a static estimate so downstream
    tooling does not confuse it with observed profiler evidence or a bounded
    compiled-executable certificate.
    """
    forward_gb: float
    ad_checkpointed_gb: float
    ad_full_gb: float
    ntff_dft_gb: float
    available_gb: float | None
    warning: str | None
    ad_segmented_gb: float | None = None
    checkpoint_every: int | None = None
    checkpoint_segments: int | None = None
    ad_active_steps: int | None = None
    ad_active_design_fraction: float | None = None
    ad_segmented_active_segments: int | None = None
    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this static estimate."""
        return "static_estimate"


    def to_dict(self) -> dict[str, float | int | None | str]:
        """Return a stable JSON-serializable AD memory artifact."""
        return {
            "evidence_class": self.evidence_class,
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
            "checkpoint_segments": self.checkpoint_segments,
            "ad_active_steps": self.ad_active_steps,
            "ad_active_design_fraction": (
                None
                if self.ad_active_design_fraction is None
                else float(self.ad_active_design_fraction)
            ),
            "ad_segmented_active_segments": self.ad_segmented_active_segments,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the estimate for research-note and CI artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


class ADMemoryComponent(NamedTuple):
    """Named contribution to a reverse-mode AD memory explanation."""

    name: str
    memory_gb: float
    share_of_selected: float
    kind: str
    unit: str | None = None
    count: int | None = None
    bytes_per_unit_gb: float | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable component artifact."""
        return {
            "name": self.name,
            "kind": self.kind,
            "memory_gb": float(self.memory_gb),
            "share_of_selected": float(self.share_of_selected),
            "unit": self.unit,
            "count": self.count,
            "bytes_per_unit_gb": (
                None
                if self.bytes_per_unit_gb is None
                else float(self.bytes_per_unit_gb)
            ),
            "explanation": self.explanation,
        }


class ADMemoryExplainabilityReport(NamedTuple):
    """Static reverse-mode AD memory explainability artifact.

    The report decomposes the selected AD estimate into named components so
    users can see whether field tape, segment-boundary carries, CPML/material
    state, or monitor state dominates the planning artifact. It is still static
    planning evidence, not profiler evidence or a bounded certificate.
    """

    n_steps: int
    strategy: str
    selected_memory_gb: float
    selected_memory_field: str
    estimate: AD_MemoryEstimate
    components: tuple[ADMemoryComponent, ...]
    dominant_component: str
    recommendations: tuple[str, ...]

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this static explanation."""
        return "static_ad_explainability"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable AD explainability artifact."""
        return {
            "evidence_class": self.evidence_class,
            "n_steps": int(self.n_steps),
            "strategy": self.strategy,
            "selected_memory_gb": float(self.selected_memory_gb),
            "selected_memory_field": self.selected_memory_field,
            "estimate": self.estimate.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "dominant_component": self.dominant_component,
            "recommendations": list(self.recommendations),
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the explanation for AD-memory diagnostics."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


class ADMemoryPlan(NamedTuple):
    """Checkpoint planning result for reverse-mode AD memory.

    ``checkpoint_every`` is the non-uniform segmented-scan chunk length.
    ``checkpoint_segments`` is the uniform segmented-scan segment count.
    ``checkpoint_mode`` is the knob selector, not a runnable-fit verdict:
    check ``full_ad_fits`` first, then require ``segmented_fits`` before wiring
    the selected segmented knob. A non-fitting plan may still carry the
    least-memory candidate knob for diagnostics.
    ``fit_safety_factor`` records the conservative multiplier used before an
    estimate is allowed to set ``full_ad_fits`` or ``segmented_fits``.
    ``evidence_class`` labels this artifact as a calibrated conservative plan,
    not a certificate or observed runtime profile.

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
    checkpoint_segments: int | None = None
    checkpoint_mode: str | None = None
    fit_safety_factor: float = 1.0
    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this conservative plan."""
        return "calibrated_conservative_plan"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable AD memory plan artifact."""
        return {
            "evidence_class": self.evidence_class,
            "n_steps": int(self.n_steps),
            "available_memory_gb": float(self.available_memory_gb),
            "target_fraction": float(self.target_fraction),
            "target_memory_gb": float(self.target_memory_gb),
            "fit_safety_factor": float(self.fit_safety_factor),
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_segments": self.checkpoint_segments,
            "checkpoint_mode": self.checkpoint_mode,
            "selected_estimate": self.selected_estimate.to_dict(),
            "full_ad_fits": bool(self.full_ad_fits),
            "segmented_fits": bool(self.segmented_fits),
            "recommendation": self.recommendation,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the plan for memory-budget and CI artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
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
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


def _nonfinite_fields(result) -> list[tuple[str, int]]:
    """Return ``[(field_name, nonfinite_count), ...]`` for the numeric array
    observables on a ``Result`` / ``ForwardResult``.

    Tracer-safe: a value under ``jax.grad`` / ``jax.jit`` tracing is an
    abstract tracer with no concrete data, so it is skipped (returns ``[]``
    rather than raising). Never raises — a divergence diagnostic must not
    itself break the return path.
    """
    import jax

    bad: list[tuple[str, int]] = []
    for name in ("time_series", "s_params"):
        arr = getattr(result, name, None)
        if arr is None:
            continue
        try:
            if isinstance(arr, jax.core.Tracer):
                continue
            a = np.asarray(arr)
        except Exception:
            continue
        if a.size == 0 or not np.issubdtype(a.dtype, np.number):
            continue
        n_bad = int(a.size - np.count_nonzero(np.isfinite(a)))
        if n_bad:
            bad.append((name, n_bad))
    return bad


_NONFINITE_CAUSE_HINT = (
    "the FDTD likely diverged. Common causes: dt above CFL, conformal=True "
    "at fine dx (a known NaN), "
    "PEC inside the CPML region, or a sub-cell PEC feature."
)


def _warn_if_nonfinite_result(result, *, context: str) -> None:
    """Emit a UserWarning (tracer-safe, never raising) when a freshly-computed
    result carries NaN/Inf observables, so an eager forward/run surfaces a
    divergence with a cause hint instead of returning silent garbage."""
    bad = _nonfinite_fields(result)
    if not bad:
        return
    detail = ", ".join(f"{n} ({c} value(s))" for n, c in bad)
    import warnings as _w

    _w.warn(
        f"[{context}] result contains non-finite values in {detail} — "
        f"{_NONFINITE_CAUSE_HINT}",
        stacklevel=3,
    )


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
    waveguide_port_flux: tuple | None = None
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

    def assert_finite(self, *, raise_on_nonfinite: bool = False) -> bool:
        """Check that the result's observables contain no NaN/Inf.

        A non-finite ``time_series`` or ``s_params`` almost always means the
        FDTD diverged rather than that the device is exotic. Returns ``True``
        when finite. With ``raise_on_nonfinite=True`` raises ``ValueError``
        instead of warning, so an automation loop can fail fast with a cause
        hint right after ``run()`` instead of propagating silent garbage into
        a downstream metric. Tracer-safe — a no-op under jax.grad/jit.

        Returns
        -------
        bool
            ``True`` if all inspected observables are finite (or unavailable
            for inspection, e.g. under tracing), ``False`` otherwise.
        """
        bad = _nonfinite_fields(self)
        if not bad:
            return True
        detail = ", ".join(f"{n} ({c} value(s))" for n, c in bad)
        msg = (
            f"Result contains non-finite values in {detail} — "
            f"{_NONFINITE_CAUSE_HINT}"
        )
        if raise_on_nonfinite:
            raise ValueError(msg)
        import warnings as _w
        _w.warn(msg, stacklevel=2)
        return False

    # ------------------------------------------------------------------
    # RF-friendly S-parameter accessors
    #
    # Convention: port numbers are 1-indexed (RF usage), so ``s(1, 1)``
    # is S11 = port1->port1.  The underlying ``s_params`` array is
    # 0-indexed with layout ``(n_ports, n_ports, n_freqs)`` — the
    # ``m, n`` ports map to ``s_params[m - 1, n - 1, :]``.
    # ------------------------------------------------------------------

    @property
    def freqs_hz(self) -> np.ndarray:
        """Frequency vector in Hz (``freqs`` is already stored in Hz).

        Returns
        -------
        (n_freqs,) float array

        Raises
        ------
        ValueError
            If this Result carries no frequency vector.
        """
        if self.freqs is None:
            raise ValueError(
                "no frequencies in this Result — run with compute_s_params=True"
            )
        return np.asarray(self.freqs)

    def _require_s_params(self) -> np.ndarray:
        """Return ``s_params`` as an ndarray or raise a clear error."""
        if self.s_params is None:
            raise ValueError(
                "no S-parameters in this Result — run with compute_s_params=True"
            )
        return np.asarray(self.s_params)

    def s(self, m: int, n: int) -> np.ndarray:
        """Complex S-parameter vector S_mn vs frequency (1-indexed ports).

        Parameters
        ----------
        m, n : int
            1-indexed port numbers. ``s(1, 1)`` is S11.

        Returns
        -------
        (n_freqs,) complex array

        Raises
        ------
        ValueError
            If this Result has no S-parameters, or if ``m``/``n`` are out
            of range (the available port count is named in the message).
        """
        sp = self._require_s_params()
        n_ports = sp.shape[0]
        if not (1 <= m <= n_ports and 1 <= n <= n_ports):
            raise ValueError(
                f"port index out of range: requested S({m},{n}) but this "
                f"Result has {n_ports} port(s) (valid 1-indexed range "
                f"1..{n_ports})"
            )
        return sp[m - 1, n - 1, :]

    def s11(self) -> np.ndarray:
        """Complex S11 vector (port1->port1). Valid for >=1 port."""
        return self.s(1, 1)

    def s21(self) -> np.ndarray:
        """Complex S21 vector (port1->port2). Valid for >=2 ports."""
        return self.s(2, 1)

    def s12(self) -> np.ndarray:
        """Complex S12 vector (port2->port1). Valid for >=2 ports."""
        return self.s(1, 2)

    def s22(self) -> np.ndarray:
        """Complex S22 vector (port2->port2). Valid for >=2 ports."""
        return self.s(2, 2)

    def s_db(self, m: int, n: int) -> np.ndarray:
        """Magnitude of S_mn in dB: ``20*log10(|S_mn|)`` (1-indexed ports).

        The magnitude is floored at ``1e-10`` before the log (matching the
        floor used by :func:`rfx.visualize.plot_s_params`) so an exact zero
        yields a large negative dB value instead of ``-inf`` and the numeric
        accessor agrees with the plotted curve at deep nulls.
        """
        mag = np.abs(self.s(m, n))
        return 20.0 * np.log10(np.maximum(mag, 1e-10))

    # ------------------------------------------------------------------
    # One-call plotting — thin wrappers over the existing engine in
    # rfx.visualize / rfx.smith. Imports stay lazy so ``import rfx``
    # remains light and headless-safe.
    # ------------------------------------------------------------------

    def plot_s_params(self, *, db: bool = True, title: str = "S-Parameters"):
        """Plot all S-parameter magnitudes vs frequency.

        Thin wrapper over :func:`rfx.visualize.plot_s_params`, which builds
        and returns its own matplotlib Figure (it does not accept an
        externally supplied Axes).

        Parameters
        ----------
        db : bool
            Plot magnitudes in dB (default) or linear.
        title : str
            Plot title.

        Returns
        -------
        matplotlib Figure

        Raises
        ------
        ValueError
            If this Result has no S-parameters.
        """
        from rfx.visualize import plot_s_params as _plot_s_params

        sp = self._require_s_params()
        return _plot_s_params(sp, self.freqs_hz, db=db, title=title)

    def plot_smith(self, *, ports: tuple[int, int] | None = None, **kw):
        """Plot an S-parameter trajectory on a Smith chart.

        Thin wrapper over :func:`rfx.smith.plot_smith`.

        Parameters
        ----------
        ports : (m, n) tuple of 1-indexed ports, optional
            Which S-parameter to plot. Defaults to ``(1, 1)`` (S11).
        **kw
            Forwarded to :func:`rfx.smith.plot_smith` (e.g. ``z0``,
            ``ax``, ``show_vswr``, ``markers``, ``title``).

        Returns
        -------
        matplotlib Axes

        Raises
        ------
        ValueError
            If this Result has no S-parameters, or ``ports`` are out of
            range.
        """
        from rfx.smith import plot_smith as _plot_smith

        if ports is None:
            m, n = 1, 1
        else:
            if len(ports) != 2:
                raise ValueError(
                    "ports must be a (m, n) tuple of two 1-indexed ports, "
                    f"got {ports!r}"
                )
            m, n = ports
        gamma = self.s(m, n)
        return _plot_smith(gamma, self.freqs_hz, **kw)

    def plot_time_series(self, *, labels=None, title: str = "Probe Time Series"):
        """Plot the probe time series.

        Thin wrapper over :func:`rfx.visualize.plot_time_series`. Requires
        ``dt`` to be present (run with ``store_dt=True``).

        Parameters
        ----------
        labels : list of str, optional
            Per-probe labels.
        title : str
            Plot title.

        Returns
        -------
        matplotlib Figure

        Raises
        ------
        ValueError
            If ``dt`` is not available in this Result.
        """
        from rfx.visualize import plot_time_series as _plot_time_series

        if self.dt is None:
            raise ValueError(
                "no dt in this Result — run with store_dt=True to plot the "
                "time series"
            )
        ts = np.asarray(self.time_series)
        if ts.ndim == 1:
            ts = ts[:, None]
        return _plot_time_series(ts, float(self.dt), labels=labels, title=title)


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


class CoaxialLineReflectionResult(NamedTuple):
    """One-port reflection from the validated coaxial transmission-line method.

    The reflection is extracted from the modal voltage ``V(z)=∫E_r dr`` sampled
    at several equally spaced reference planes on a real coax line terminated in
    a matched resistive feed (see ``compute_coaxial_line_reflection``). The
    complex propagation constant ``gamma`` is self-measured (matrix pencil), so
    the result is Z0-free and immune to the coarse-mesh ``|V/I|`` bias.

    ``recurrence_residual`` is the per-frequency single-TEM-mode validity gate
    (0 = a clean two-wave field). ``annulus_cells`` is the resolution metric
    ``(outer-inner)/dx``; below ~3.5 cells the mode is under-resolved and the
    high-frequency reflection degrades. ``status`` is ``"passed"``,
    ``"under_resolved"`` (annulus too coarse), or ``"contaminated"`` (recurrence
    residual exceeded the gate at one or more frequencies).
    """

    s11: np.ndarray
    freqs: np.ndarray
    gamma: np.ndarray
    recurrence_residual: np.ndarray
    fit_residual: np.ndarray
    annulus_cells: float
    z0_numerical_ohm: np.ndarray
    termination: str
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
    n_probes: int = 5
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
    "ADMemoryComponent",
    "ADMemoryExplainabilityReport",
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
    "CoaxialLineReflectionResult",
    "_MSLPortEntry",
    "MSLSMatrixResult",
]
