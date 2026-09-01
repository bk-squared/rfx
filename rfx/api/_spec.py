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
from typing import Mapping, NamedTuple

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


def _artifact_to_dict(value: object) -> object:
    """Serialize nested report artifacts without importing non-leaf rfx modules."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _artifact_to_dict(value.to_dict())
    if hasattr(value, "to_json") and callable(value.to_json):
        return _artifact_to_dict(json.loads(value.to_json()))
    if hasattr(value, "tolist") and callable(value.tolist):
        return _artifact_to_dict(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _artifact_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_artifact_to_dict(item) for item in value]
    if isinstance(value, list):
        return [_artifact_to_dict(item) for item in value]
    raise TypeError(
        f"preflight nested artifact contains unsupported value {type(value).__name__}"
    )


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
    ``checkpoint_segments`` on the uniform segmented-scan path. The
    segmented model counts segment-boundary storage (carry + cotangent,
    ``2 × active_segments``) PLUS the live-segment rematerialization tape
    (one segment's per-step field tape resident during backward replay,
    issue #277); it is minimized near ``sqrt(2 × n_steps)`` steps per
    segment. When present,
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
    ad_segmented_active_segments: int | None = None
    #: Which grid the numbers describe (#696): ``"uniform"`` or
    #: ``"nonuniform"``. The estimate used to be derived from a private
    #: re-derivation of the shape that matched NEITHER grid wherever
    #: per-face pads or 2-D mode changed it, and nothing in the artifact
    #: said which grid it meant — a uniform-lane ``dt`` has already been
    #: mistaken for the NU one in this repo.
    grid_kind: str | None = None
    #: ``"built"`` when the shape came from the grid the solve will
    #: build, ``"estimated_from_domain"`` when the grid build failed and
    #: the legacy domain arithmetic was used as a fallback.
    grid_source: str | None = None
    grid_shape: tuple[int, int, int] | None = None
    #: Surface-impedance (f0) sheet operator state — three boolean
    #: tangential edge masks plus ``sigma_sheet`` (#677). Zero when no f0
    #: sheet is registered. It was counted NOWHERE before #696, so a
    #: lossy board estimated identically to the same board without loss.
    sheet_gb: float | None = None
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
            "ad_segmented_active_segments": self.ad_segmented_active_segments,
            "grid_kind": self.grid_kind,
            "grid_source": self.grid_source,
            "grid_shape": (
                None if self.grid_shape is None
                else [int(v) for v in self.grid_shape]
            ),
            "sheet_gb": (
                None if self.sheet_gb is None else float(self.sheet_gb)
            ),
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


class ADMemoryActionHint(NamedTuple):
    """Actionable AD-memory preflight hint derived from planning evidence."""

    code: str
    severity: str
    message: str
    action: str
    checkpoint_mode: str | None = None
    checkpoint_every: int | None = None
    checkpoint_segments: int | None = None
    blocking: bool = False

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this static action hint."""
        return "static_action_hint"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable action-hint artifact."""
        return {
            "evidence_class": self.evidence_class,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "action": self.action,
            "checkpoint_mode": self.checkpoint_mode,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_segments": self.checkpoint_segments,
            "blocking": bool(self.blocking),
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the action hint with non-finite floats rejected."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


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


class ADMemoryPreflightReport(NamedTuple):
    """Composite AD-memory preflight artifact.

    This report composes static memory planning, static explainability,
    optional mesh advisories, and optional trace-time JAX saved-residual
    diagnostics. It is not runtime profiler evidence, XLA memory analysis,
    a peak-memory guarantee, a certificate, or RF validation.
    """

    n_steps: int
    available_memory_gb: float
    target_fraction: float
    target_memory_gb: float
    fit_safety_factor: float
    status: str
    supported_checkpoint_mode: str | None
    checkpoint_every: int | None
    checkpoint_segments: int | None
    full_ad_fits: bool
    checkpointing_fits: bool
    memory_plan: ADMemoryPlan
    explainability: ADMemoryExplainabilityReport
    mesh_report: MeshIntelligenceReport | None
    residual_diagnostic: object | None
    action_hints: tuple[ADMemoryActionHint, ...]
    evidence_boundaries: tuple[str, ...]
    recommendation: str

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this composite preflight."""
        return "composite_ad_memory_preflight"

    @property
    def source_evidence_classes(self) -> tuple[str, ...]:
        """Evidence classes from nested artifacts, in stable order."""
        classes: list[str] = [
            self.memory_plan.evidence_class,
            self.memory_plan.selected_estimate.evidence_class,
            self.explainability.evidence_class,
        ]
        residual_source = getattr(
            self.residual_diagnostic,
            "source_evidence_class",
            None,
        )
        if residual_source is not None:
            classes.append(str(residual_source))
        residual_evidence = getattr(self.residual_diagnostic, "evidence_class", None)
        if residual_evidence is not None:
            classes.append(str(residual_evidence))

        deduped: list[str] = []
        for evidence_class in classes:
            if evidence_class not in deduped:
                deduped.append(evidence_class)
        return tuple(deduped)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable AD-memory preflight artifact."""
        return {
            "evidence_class": self.evidence_class,
            "source_evidence_classes": list(self.source_evidence_classes),
            "n_steps": int(self.n_steps),
            "available_memory_gb": float(self.available_memory_gb),
            "target_fraction": float(self.target_fraction),
            "target_memory_gb": float(self.target_memory_gb),
            "fit_safety_factor": float(self.fit_safety_factor),
            "status": self.status,
            "supported_checkpoint_mode": self.supported_checkpoint_mode,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_segments": self.checkpoint_segments,
            "full_ad_fits": bool(self.full_ad_fits),
            "checkpointing_fits": bool(self.checkpointing_fits),
            "memory_plan": self.memory_plan.to_dict(),
            "explainability": self.explainability.to_dict(),
            "mesh_report": _artifact_to_dict(self.mesh_report),
            "residual_diagnostic": _artifact_to_dict(self.residual_diagnostic),
            "action_hints": [hint.to_dict() for hint in self.action_hints],
            "evidence_boundaries": list(self.evidence_boundaries),
            "recommendation": self.recommendation,
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the preflight report with non-finite floats rejected."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        options["allow_nan"] = False
        return json.dumps(self.to_dict(), **options)


class ADCompiledMemoryCertificate(NamedTuple):
    """Bounded compiler-memory certificate for one exact compiled executable.

    The report is derived from a caller-supplied JAX compiled object's
    ``memory_analysis()`` result plus complete exact-scope metadata. It is not a
    universal peak-memory predictor, runtime profiler artifact, RF validation
    result, or proof that source/config digests correspond to an opaque
    executable.
    """

    status: str
    status_reason: str
    available_memory_gb: float
    target_fraction: float
    target_memory_gb: float
    compiler_reported_required_bytes: int | None
    compiler_reported_required_gb: float | None
    temp_size_in_bytes: int | None
    argument_size_in_bytes: int | None
    output_size_in_bytes: int | None
    alias_size_in_bytes: int | None
    temp_gb: float | None
    argument_gb: float | None
    output_gb: float | None
    alias_gb: float | None
    exact_scope: Mapping[str, object] | None
    scope_status: str
    scope_status_reason: str
    scope_digest: str | None
    config_digest: str | None
    environment_digest: str | None
    memory_analysis_status: str
    memory_analysis_status_reason: str
    jax_version: str
    evidence_boundaries: tuple[str, ...]
    recommendations: tuple[str, ...]
    source_preflight: object | None = None

    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this bounded certificate."""
        return "bounded_certificate"

    @property
    def is_valid_certificate(self) -> bool:
        """Whether compiler evidence and exact scope are complete."""
        return self.status in {
            "compiler_estimate_within_budget",
            "compiler_estimate_exceeds_budget",
        }

    @property
    def estimate_within_budget(self) -> bool | None:
        """Whether the JAX compiler memory *estimate* fits the target budget.

        This reflects ``Compiled.memory_analysis()``, a compiler estimate — not
        a runtime guarantee. It does not model allocator fragmentation or
        runtime scratch, so ``True`` does not promise the run avoids OOM.
        Returns ``None`` when compiler evidence or exact scope is incomplete.
        """
        if self.status == "compiler_estimate_within_budget":
            return True
        if self.status == "compiler_estimate_exceeds_budget":
            return False
        return None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable bounded-certificate artifact."""
        return {
            "evidence_class": self.evidence_class,
            "status": self.status,
            "status_reason": self.status_reason,
            "is_valid_certificate": self.is_valid_certificate,
            "estimate_within_budget": self.estimate_within_budget,
            "available_memory_gb": float(self.available_memory_gb),
            "target_fraction": float(self.target_fraction),
            "target_memory_gb": float(self.target_memory_gb),
            "compiler_reported_required_bytes": self.compiler_reported_required_bytes,
            "compiler_reported_required_gb": self.compiler_reported_required_gb,
            "temp_size_in_bytes": self.temp_size_in_bytes,
            "argument_size_in_bytes": self.argument_size_in_bytes,
            "output_size_in_bytes": self.output_size_in_bytes,
            "alias_size_in_bytes": self.alias_size_in_bytes,
            "temp_gb": self.temp_gb,
            "argument_gb": self.argument_gb,
            "output_gb": self.output_gb,
            "alias_gb": self.alias_gb,
            "exact_scope": _artifact_to_dict(self.exact_scope),
            "scope_status": self.scope_status,
            "scope_status_reason": self.scope_status_reason,
            "scope_digest": self.scope_digest,
            "config_digest": self.config_digest,
            "environment_digest": self.environment_digest,
            "memory_analysis_status": self.memory_analysis_status,
            "memory_analysis_status_reason": self.memory_analysis_status_reason,
            "jax_version": self.jax_version,
            "evidence_boundaries": list(self.evidence_boundaries),
            "recommendations": list(self.recommendations),
            "source_preflight": _artifact_to_dict(self.source_preflight),
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the certificate with non-finite floats rejected."""
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


def _auto_source_decay_time(fr, waveform=None) -> float:
    """Auto Harminv window start: twice the source pulse-completion scale.

    The historical hardcode was ``2.0*3.0*tau`` — the cutoff=3
    ``GaussianPulse`` envelope (``t0 = 3*tau``). A waveform exposing its
    own ``t0`` (``GaussianPulse``/``ModulatedGaussian``:
    ``t0 = cutoff*tau``) makes the window scale with the actual onset —
    ``2*t0`` = ``9*tau`` at cutoff=4.5 — instead of assuming the cutoff=3
    envelope (post-#392 review). Without a waveform this returns
    ``2.0*(3.0*tau)``, bitwise-identical to the historical
    ``2.0*3.0*tau`` (scaling by 2.0 is exact in binary floating point),
    with ``tau`` derived from the frequency range at the historical
    bw=0.8.
    """
    f_center = (fr[0] + fr[1]) / 2
    bw = 0.8
    tau = 1.0 / (f_center * bw * np.pi)
    return 2.0 * float(getattr(waveform, "t0", 3.0 * tau))


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
                         source_decay_time=None, bandpass=None,
                         waveform=None):
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
        waveform : source waveform or None
            Used only when ``source_decay_time`` is None: when the
            waveform exposes ``t0`` (``GaussianPulse`` /
            ``ModulatedGaussian``: ``t0 = cutoff*tau``), the auto window
            starts at ``2*t0`` so a longer-onset pulse (e.g.
            ``cutoff=4.5`` → ``9*tau``) is fully skipped. Without a
            waveform the historical cutoff=3 envelope (``2*(3*tau)``) is
            used, bitwise-identical to the previous hardcode.

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
            source_decay_time = _auto_source_decay_time(fr, waveform)

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
    # Issue #706: opt-in two-plane realization for a PEC body that fills
    # exactly one cell along its normal axis. Default False = the
    # load-bearing one-plane behaviour (#677-validated).
    two_plane: bool = False


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
    # Opt-in reference-plane port waves for the wire S-matrix OFF-diagonal
    # extraction (issue #313): when set to an integer N, the production
    # scan registers TWO line V/I reference planes at N and 2N cells
    # outboard (into the DUT) and the off-diagonal S_ij switch to plane
    # waves with measured Zc/beta. None (default) = shipped port-cell
    # behaviour, byte-identical. Diagonal S_jj always stays on the legacy
    # path either way.
    reference_plane_cells: int | None = None
    # Soft-source amplitude semantics (issue #571, option 4):
    # 'field' | 'current' | None (= legacy per-path default, deprecated).
    # Only meaningful when impedance == 0.0 (add_source soft sources); port
    # entries (impedance > 0) keep their own port-normalized waveform
    # contract and never set this. Defaulted so every non-add_source
    # _PortEntry construction site is untouched.
    amplitude_kind: str | None = None


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
    method: str = "bloch"


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
    """Waveguide scattering result assembled one driven port at a time.

    ``settling_db`` (issue #538) is the per-driven-run energy ring-down
    witness: worst end/peak tail ratio over ALL FOUR recorded per-port
    time series (``v_probe_t``/``v_ref_t``/``i_probe_t``/``i_ref_t`` —
    single-series witnesses measured up to 8.3 dB optimistic vs the
    records the extraction actually consumes) for each driven run, in dB;
    two-run variants (flux/normalized) take the worst of the device AND
    reference runs. Rule: below −40 dB, same ``_SETTLING_WITNESS_DB``
    threshold and aggregate warning as the lumped/MSL path. Scope limits
    (modal records only; peak includes the incident pulse): see
    ``settling_db_from_port_records``. ``None`` on the one path that has
    not adopted the witness: the uniform multimode branch
    (``extract_multimode_s_matrix*``) — tracked on the issue. The NU
    sibling (``_compute_waveguide_s_matrix_nu``) carries it since the
    issue #827 waveguide-instance fix. NaN
    entries mean the run was traced (AD path) and the host-side witness
    was skipped rather than concretised.
    """
    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    port_directions: tuple[str, ...]
    reference_planes: np.ndarray
    settling_db: np.ndarray | None = None


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


@dataclass
class CoaxialTwoPortResult:
    """Two-drive coaxial 2-port S-parameters on a through line (issue #489 stage 2).

    STATUS: **VALIDATED WITH SCOPE** (issue #489, PI decision 2026-08-06 —
    ``docs/guides/sparameter_support_matrix.md``, the S-parameter-family
    companion where this row lives — see also
    ``docs/guides/sparameter_support_matrix.json``, its machine-readable
    twin). The EXPERIMENTAL label held through three legs closing in
    sequence — wiring pin, mesh-refinement convergence witness, and an
    ``eps_scale`` AD gate — and lifted once a fourth, an external referee,
    also closed. Evidence chain: an external openEMS referee (crossval 21,
    ``validation/crossval/21_coax_two_port_referee.py``, VESSL run-3
    ``369367251629`` and the first default-scale green promoted-lane run
    VESSL ``369367252220``) brackets — it does not judge — this method's own
    ``|S21|`` on the through-line class, and, via the port's own measured
    ``beta`` (not an idealized analytic one), its phase; a mesh-refinement
    convergence witness (VESSL ``369367251845``) moved the measured/analytic
    ``beta`` ratio from ``1.1208`` to ``1.0662`` (implied convergence order
    ``p ~= 1.5``, two-point, from a single 1.5x step); the ``eps_scale`` AD
    channel below is ``GRAD_SAFE`` (``tests/test_ad_surface_contract.py``);
    and this method's own reciprocity (``0.3%`` magnitude / ``0.21`` degree
    phase) and ``cond(A) <= 1.11`` are measured on the committed fixture
    (below). This extends the validated 1-port
    coax-line method (:meth:`compute_coaxial_line_reflection`) to two ports by
    building a single through line with a matched annular-resistor feed near
    EACH z end, driving each end's own TEM TFSF source in turn (two separate
    FDTD runs), and recovering each port's own forward/back wave amplitudes
    (not assuming the non-driven port sees zero incident wave — see
    :func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes` for
    why the naive ``S[j,i]=b_j/a_i`` ratio has a hard terminator-reflection
    floor that this two-drive solve removes).

    **Scope limitation that must not be silently dropped — NOT covered by
    the evidence chain above**: every DUT this method can currently gate
    against (none / a matched feed / a coaxial dielectric plug) is
    azimuthally symmetric and excites only TM0n modes. Issue #489 exists
    for TRANSITION designers whose discontinuities excite TE11 (cutoff
    25.17 GHz on the validated SMA line — inside the 4-12 GHz band's
    evanescent tail, surviving to the first probe plane at roughly 0.10 of
    its launch amplitude). A battery built only on symmetric DUTs certifies
    a class that EXCLUDES the target class — coax<->planar transitions are
    the SEPARATE :meth:`compute_coax_msl_transition` lane, which stays
    **EXPERIMENTAL, diagnostic-only** regardless of this promotion (its own
    reciprocity is unmeasured pending a settled run — see that method's own
    docstring). Nor does this evidence generalize beyond this single
    through-line coax geometry family (SMA-class, the committed fixture's
    own dimensions — ``a=0.635mm``, ``b=2.055mm``, PTFE ``eps_r=2.1``) to
    other coax geometries.

    ``s_params[j, i, :]`` is the response measured at port ``j`` while port
    ``i`` is driven (standard S-matrix convention), referenced to each port's
    OWN reference plane (its feed resistor's axial plane — see
    ``reference_planes``). ``cond_a`` is the per-frequency condition number of
    the incident-wave matrix inverted by the two-drive solve: it bounds
    DEGENERACY of the two drives only, and is blind to a systematic
    incident/outgoing mislabel at one port (see the cited function's
    docstring) — passivity (checked downstream by
    ``_warn_if_nonpassive_smatrix`` via ``_finalize_sparam_result``, NOT
    bypassed here) is the only handle on that defect family.
    ``recurrence_residual`` / ``fit_residual`` / ``gamma`` are per (port
    array, drive, frequency) — recurrence_residual 0 means a clean
    single-TEM-mode field at that array during that drive; ``gamma`` is that
    array's own matrix-pencil-fitted complex propagation constant (Z0-free,
    from local probes only). **Measured 2026-08-02: a given array's
    ``Re(gamma)`` is NOT independent of which drive produced it** — "own
    drive" (that array's own source active) and "other drive" (that array
    receiving the transmitted signal) give substantially different, but each
    internally self-consistent (agreeing between the two mirror-symmetric
    arrays to 2-5%), estimates; see
    :func:`_assemble_coaxial_two_port_from_voltages` for the full finding and
    ``tests/test_coax_two_port_fdtd.py::
    test_matched_through_line_transmits_reciprocally`` for the mechanism
    check this motivated. Do not read a single array's ``gamma`` as *the*
    line attenuation without averaging across all 4 (2 arrays x 2 drives)
    measurements. ``annulus_cells`` is the shared resolution metric (same
    convention as the 1-port method, below ~3.5 cells is under-resolved).
    ``settling_db`` is a per-drive ring-down witness (worst end/peak E^2 ratio,
    dB, over one point probe per array — same convention as the MSL/mixed
    lanes; above -40 dB suggests the fixed-length record may have been
    truncated before the structure rang down). Since issue #662 that bar is
    ENFORCED, not just documented: a violating drive emits a
    ``UserWarning`` naming the drive and its measured value, so a truncated
    record can no longer return a plausible-looking ``s_params`` in silence.
    It stays a warning, never an exception — short diagnostic runs are a
    legitimate use of this method.

    **``eps_scale`` (differentiable) path** (:meth:`compute_coaxial_two_port`'s
    own ``eps_scale`` parameter, issue #489 leg 3): ``status`` takes a FOURTH
    value, ``"differentiable"``, in place of ``"passed"``/``"contaminated"``
    — the traced ``rec_resid`` cannot be Python-branched on to distinguish
    those two, so ``"differentiable"`` means only "geometry-resolved"
    (``annulus_cells >= 3.5``; below that it is still ``"under_resolved"``),
    NOT "fit-clean"; inspect ``recurrence_residual`` directly if the
    contamination signal matters. ``settling_db`` stays ``nan`` for both
    drives on this path (the ring-down witness needs a concrete time series).
    ``cond_a`` is still returned as a traced value, but the ill-conditioning
    WARNING ``cond_warn`` normally controls does not fire — see
    :meth:`compute_coaxial_two_port`'s own docstring for why.

    **Numerical line attenuation, not just a reflection artifact**: on the
    validated 60 mm / 40 GHz fixture, the discrete (3.79-cell annulus)
    through line itself attenuates the transmitted wave — measured
    ``|S21|`` 0.96 (4 GHz) down to 0.74 (12 GHz) even with ``|S11|`` <= 0.05
    throughout. A post-hoc consistency check (run after this measurement,
    not predeclared) compares ``|S21|`` against the independently
    matrix-pencil-fitted ``exp(-Re(gamma) * L12)`` (see ``gamma`` above): the
    ``|S21|`` deficit equals what the local decay-rate fits predict over the
    reported port separation. **What this check catches and what it does
    not**: it is sensitive to SCALE-type deficits (amplitude
    mis-normalization, mode conversion, a bad wave split — ``gamma`` is
    fit from the field's shape along z, not its absolute scale, so a scale
    bug in ``|S21|`` would not be echoed by a matching shift in the fitted
    ``gamma``). It is structurally BLIND to reference-plane referral errors:
    a referral error ``delta`` at either plane scales the wave amplitude by
    ``exp(+/-gamma*delta)`` while ``L12`` grows by the same ``delta``, so the
    compensation factor absorbs a referral error exactly (verified to five
    decimal places even at +30 cells of injected error) — a wrong reference
    plane passes this check unchanged. The under-resolved-annulus recipe
    (>=4 cells) that this repo already documents for reflection accuracy
    (``compute_coaxial_line_reflection``) applies to TRANSMISSION magnitude
    here too, even when ``status`` reports ``"passed"``.
    """

    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    reference_planes: np.ndarray
    cond_a: np.ndarray
    recurrence_residual: np.ndarray
    fit_residual: np.ndarray
    gamma: np.ndarray
    annulus_cells: float
    settling_db: np.ndarray
    status: str
    flux_monitors: dict | None = None


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
        Characteristic impedance extracted via the N-probe SVD
        least-squares wave-decomposition fit (issue #80 Fix C),
        per driven-port run (``Z0[i, :]`` is from run with port i driven).
    beta : (n_freqs,) complex
        Propagation constant β from the N-probe least-squares fit
        (issue #80 Fix C) at the first port's run.
    reliable : (n_ports, n_freqs) bool, optional
        Per-port wave-split reliability. False marks standing-wave-null bins
        where both voltage and current collapse below 10% of their band
        medians. S values at those bins are retained unchanged.

        ``reliable[p, k]`` is False when PORT ``p``'s probe plane collapsed
        at bin ``k`` in AT LEAST ONE drive.  Every ``(driven, port)`` record
        the solve consumes is covered, not only the own-drive diagonal
        (issue #522) — a collapse at a *passive* port's plane during
        someone else's drive used to be invisible here while still
        corrupting the result.

        *What a False entry condemns*: the ENTIRE frequency slice
        ``S[:, :, k]``, not just the column ``S[:, p, k]`` the pre-#507
        single-ratio assembly confined it to.  ``S`` is ``B·A⁻¹`` over all
        drives, so one collapsed wave pair contaminates the whole slice.
        Drop the bin; the index tells you which plane to investigate.

        *What a True entry does not certify*: accuracy.  It means the
        low-signal threshold did not fire, nothing more.

        *Cost of the widened coverage, measured*: the threshold is relative
        to each record's OWN band median, so a port sitting in a deep
        stopband is not flagged wholesale — but individual deep bins ARE
        flagged.  Live extractor runs on the two filter geometries flagged 2
        bins of 100 on the ``msl_notch_e4`` fixture and 12 of 120 on the
        Sheen LPF leg (``validation/crossval/07_sheen_lpf.py`` at its
        ``--n-freqs`` default), and the notch fixture's two ARE the notch
        centre — 3.6273 GHz, which the committed fixture meta records at
        −30.66 dB.  The two COUNTS are not recomputable from the committed
        JSON: those fixtures store S magnitudes only, with no V/I dump, so
        checking them means re-running the extractor and reading
        ``reliable``.  That is not a false alarm
        — at a −30 dB notch the passive port's wave split really is
        low-signal and the extractor cannot certify the depth — but a filter
        user loses exactly the bin they care about and should read the depth
        from ``S_raw`` or the flux channel with that caveat.

        ``np.all(reliable, axis=0)`` is therefore the right per-bin screen:
        it keeps exactly the bins where no plane the solve reads had
        collapsed.
    settling_db : (n_ports,) float, optional
        Ring-down settling witness per driven-port run: the WORST (largest)
        over ALL port probe planes of ``10*log10(mean Ez^2 over the last 10%
        of the record / peak Ez^2)``. Multiple planes per port are sampled
        because a single plane is standing-wave-node sensitive — measured
        18.1 dB spread across planes on the same under-settled record, i.e.
        a one-point witness can PASS at a node while the record is hot. Values above −40 dB mean the
        fixed-length record was truncated before the structure rang down, and
        the DFT-derived S-parameters of that run are suspect (measured on the
        Sheen-1990 LPF: num_periods=20 left the stopband ring unsettled and
        produced |S| column-power poles up to ~1.8e3 that shrank monotonically
        with record length). Compare against the project's −40 dB ring-down
        settling rule (docs/guides/simulation_methodology.md) before quoting
        any S value from this result.
    S_raw : (n_ports, n_ports, n_freqs) complex, optional
        The S-matrix exactly as extracted, BEFORE passivity projection.
        Stored whenever the projection changed anything, so no information
        is discarded by enforcing the bound.
    passivity_correction : (n_freqs,) float, optional
        Per-frequency amount clipped by the passivity projection:
        ``max(sigma_max(S_raw(f)) - 1, 0)``. Zero where the extraction was
        already passive. This is the honesty metric — a bin with a large
        correction is a measurement artifact (check ``reliable`` and
        ``settling_db`` for the cause), and its projected value inherits
        that uncertainty.
    port_names : tuple[str, ...]
    assembly : str, optional
        Which rule produced ``S`` — ``"multi_drive_solve"`` (normal:
        ``S = B·A⁻¹`` over all drives, issue #507) or
        ``"single_ratio_fallback"``. The fallback is taken when the solve
        returns non-finite entries on a degenerate drive system; it is the
        SUPERSEDED per-column rule ``S[j, d] = b_j / a_d``, which reports a
        passive port's echo as the driven port's own reflection whenever
        that port is not matched.

        **Read this before trusting a fallback result.** The fallback's
        characteristic symptom is column power above 1, and with the default
        ``enforce_passivity=True`` that symptom is clipped out of ``S`` — but
        it is not erased from the result: ``passivity_correction`` records
        how much was clipped and ``S_raw`` keeps the unprojected matrix, and
        the run also emits both a fallback warning and a passivity-guard
        warning. So a fallback is not silent; this field is simply the
        *specific* signal. Column power above 1 has several causes (an
        under-settled record, a standing-wave null, a mis-scaled current) and
        only one of them is the fallback — that is what this field
        disambiguates. ``None`` while tracing (the finiteness test cannot run
        on a tracer, so the solve result is taken as-is — see ``cond_a``).
    cond_a : (n_freqs,) float, optional
        Per-frequency condition number of the drive matrix ``A``. Bounds
        DEGENERACY of the drive system only — it is **not** a reliability
        or accuracy score, and a low value does not certify the result
        (same contract as the coax lane's
        :func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`).
        ``None`` while tracing.
    beta_railed : (n_ports, n_freqs) bool, optional
        Per-port β-scan rail mask for the FITTED numbers (issue #681).
        ``beta_railed[p, k]`` is True when port *p*'s own-drive N-probe β
        scan failed to bracket its residual optimum at bin *k* — the raw
        argmin sat at an edge node of the ±35% window around the analytic
        Hammerstad-Jensen guess, or the refined β landed within half a
        grid step of a window limit.  ``Z0[p, k]`` (and ``beta[k]`` for
        ``p = 0``) at such a bin is the scan-window limit, NOT a
        measurement — do not quote it.  ``S`` is NOT condemned: S11/S21
        ride on the analytic Z0 anchor, never on the fitted β.  The
        own-drive diagonal is exactly the provenance of every fitted
        number this result carries (``Z0[i, :]`` comes from port *i*'s
        own driven run).  ``None`` while tracing.
    """
    S: np.ndarray
    freqs: np.ndarray
    Z0: np.ndarray
    beta: np.ndarray
    port_names: tuple[str, ...] = ()
    reliable: np.ndarray | None = None
    settling_db: np.ndarray | None = None
    S_raw: np.ndarray | None = None
    passivity_correction: np.ndarray | None = None
    assembly: str | None = None
    cond_a: np.ndarray | None = None
    beta_railed: np.ndarray | None = None


@dataclass
class MixedSMatrixResult:
    """Mixed-family S-matrix result (issue #488, lumped/wire + MSL v1).

    Unlike the per-family extractors, ``S`` here is in the **Kurokawa
    power-wave convention** (every wave amplitude divided by
    ``sqrt(Re(Z0_port))``): with unequal reference impedances across
    families a pseudo-wave ``b/a`` ratio is off by ``sqrt(Z_j/Z_i)``
    (issue #460), so the mixed lane normalizes to power waves by
    construction. Off-diagonal PHASE across families is provisional —
    the lumped/wire cell and the MSL de-embedded plane are different
    reference-plane conventions (magnitude is the validated observable).

    Attributes
    ----------
    S : (n_ports, n_ports, n_freqs) complex
        Power-wave S-matrix; ``S[i, j]`` = response at port *i* driving
        port *j*. Port order: lumped/wire ports (registration order),
        then MSL ports (registration order).
    freqs : (n_freqs,) float
        Frequency grid in Hz.
    port_names : tuple of str
        Combined port names (lumped/wire ports are auto-named
        ``"lw<k>"``; MSL ports keep their registered names).
    port_families : tuple of str
        ``"lumped"`` / ``"wire"`` / ``"msl"`` per port, same order.
    z0_ref : (n_ports,) float
        Power-wave reference impedance per port: the registered port
        impedance for lumped/wire, analytic Hammerstad-Jensen Z0 for MSL.
    settling_db : (n_ports,) float, optional
        Ring-down settling witness per driven-port run (worst end/peak
        Ez^2 over the MSL probe planes, dB; above -40 dB = truncation
        suspect, same convention as :class:`MSLSMatrixResult`).
    s21_power_witness : np.ndarray | None
        (n_msl, n_lw, n_freqs) real — extractor-independent |S21|
        cross-check for lumped/wire-driven columns: the MSL arriving
        power-wave magnitude over ``|a_drive|`` reconstructed from the
        delivered power ``P_del = 0.5*Re(Z_in)*|I|^2`` and
        ``(1 - |S_jj|^2)`` (does not trust the port-cell a-wave
        magnitude; issue #313 triangulation).
    reliable : np.ndarray | None
        (n_msl, n_freqs) bool — MSL standing-wave-null reliability mask
        from each MSL port's own driven run (False = ill-conditioned bin).
    beta_railed : np.ndarray | None
        (n_msl, n_freqs) bool — β-scan rail mask from each MSL port's
        own driven run, same criterion as
        :attr:`MSLSMatrixResult.beta_railed` (issue #681).  In this lane
        the N-probe fit is DIAGNOSTIC ONLY (the |Zc| deviation warning);
        a True bin means that diagnostic's β/|Zc| are the scan-window
        limit, not a measurement.  ``S`` is unaffected.
    S_raw / passivity_correction :
        As in :class:`MSLSMatrixResult` — set only when the passivity
        projection touched at least one bin.
    """
    S: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    port_families: tuple[str, ...]
    z0_ref: np.ndarray
    settling_db: np.ndarray | None = None
    s21_power_witness: np.ndarray | None = None
    reliable: np.ndarray | None = None
    S_raw: np.ndarray | None = None
    passivity_correction: np.ndarray | None = None
    beta_railed: np.ndarray | None = None
    # Arch-A magnitude channel (issue #488): off-diagonal magnitudes come
    # from Poynting flux by default; ``S_wave`` keeps the raw power-wave
    # matrix for comparison (None when magnitude_channel="wave").
    S_wave: np.ndarray | None = None
    magnitude_channel: str = "wave"


@dataclass
class CoaxMSLTransitionResult:
    """EXPERIMENTAL two-drive coax<->microstrip transition S-parameters (issue #489 leg 4).

    STATUS: **EXPERIMENTAL — not in the validated set**
    (``docs/guides/sparameter_support_matrix.md`` / ``.json``). This is the
    coax<->planar generalization of the #488 mixed-family lane
    (:class:`MixedSMatrixResult`) and reuses the #489 stage-2 coax two-port
    machinery's per-port wave extraction
    (:func:`rfx.sources.coaxial_port.coaxial_line_reflection_from_plane_voltages`)
    for BOTH ports — including the MSL side, in place of the diagnostic-only,
    sign-unstable N-probe SVD fit (``extract_msl_nprobe``; see its own
    docstring and :meth:`_SparamMixin.compute_mixed_s_matrix`'s "N-probe line
    Zc: DIAGNOSTIC ONLY" comment for the documented branch-sign instability
    this sidesteps) — a deliberate deviation from imitating the #488 lane's
    own MSL extractor choice, made because the coax matrix-pencil fit
    deterministically pins the propagation-constant branch
    (``beta = Im(gamma) > 0``) while the N-probe fit does not.

    Like :class:`MixedSMatrixResult`, ``s_params`` is in the **Kurokawa
    power-wave convention**: each port's raw modal-voltage wave amplitude
    (``forward_amp``/``backward_amp``, both in **volts** — the matrix-pencil
    fit is Z0-free) is divided by ``sqrt(z0_ref)`` of that port's OWN
    reference impedance (real reference impedances only — no ``Re()`` is
    taken anywhere in this lane's assembly; see
    :func:`_assemble_coax_msl_transition_from_voltages`) BEFORE the
    two-drive solve
    (:func:`rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes`).
    This step is load-bearing here in a way it was not for the coax-coax
    two-port lane: solving directly on RAW (un-normalized) volt-wave
    amplitudes leaves S_ii (diagonal) exactly correct but scales each
    off-diagonal S_ij (i != j) by ``sqrt(Z0_i/Z0_j)`` relative to the true
    power-wave value — invisible on a coax-coax through line (equal Z0
    cancels) but NOT invisible here, since the coax port's analytic TEM Z0
    and the MSL port's Hammerstad-Jensen Zc are generally different. This is
    the pre-declared "impedance-convention mismatch" failure mode: getting
    this normalization wrong presents as spurious reflection/transmission
    error even on an ideal, lossless transition.

    Reference planes sit at the physical launch discontinuity on BOTH sides
    by construction (``junction_z`` on the coax's own z axis, ``junction_x``
    on the MSL's own x axis — see ``reference_planes``), specifically to
    minimize the OTHER pre-declared failure mode ("reference-plane
    mismatch": the coax's axial z-feed-plane convention has no direct
    analogue in the MSL's along-trace x reference plane — these are
    different geometric axes). Off-diagonal PHASE nonetheless mixes two
    different transverse-mode conventions (coax TEM radial E-integral vs MSL
    quasi-TEM z line-integral) and inherits #488's own caveat: magnitude is
    the validated observable, phase is provisional.

    NOT_TRACEABLE (inherits #488's AD scope): both this result's own
    assembly and the underlying two extractors it composes
    (:meth:`_SparamMixin.compute_coax_msl_transition`'s coax stub +
    :func:`rfx.sources.msl_port.make_msl_port_sources`) run a concrete NumPy
    path only. AD is explicitly out of scope for this leg (unlike stage-2
    coax-coax, which grew an ``eps_scale`` channel in PR #572 — no
    equivalent channel exists here).

    On the one committed fixture (issue #581 adversarial review, findings
    B2/B3): reciprocity and the two-drive solve's raw ``cond_a`` are badly
    degenerate. The ORIGINAL attribution written for this fixture —
    "near-degenerate two-drive amplification from strong junction
    reflection" — did not survive its own data and was retracted. Three
    independent checks on this fixture refute it: (i) ``cond_a`` is almost
    entirely a per-drive amplitude SCALE artifact, not geometric
    near-parallelism — after per-column equilibration (see
    ``cond_a_equilibrated`` below) the two drives' incident-wave columns
    are near-ORTHOGONAL, not nearly parallel; (ii) the "both ports strongly
    reflecting" premise fails on this fixture's own measured ``|S22|``
    (near 0 at two of three bins, not near 1); (iii) the signature that
    DOES match is a drive-amplitude mismatch between the two unrelated
    source constructions (coax TEM plane source vs MSL Ez injection) —
    5-9 orders of magnitude apart on this fixture, visible directly in
    ``a_inc`` (see below). The PREDECLARED alternative — an MSL wave-
    extraction instrument-scoping limit, not junction physics — is
    positively supported instead: the MSL probe ladder on this fixture
    spans only 0.34%-3.37% of the guided wavelength across the three
    measured frequencies, and the fitted ``gamma`` on the MSL array does
    not track the analytic Hammerstad-Jensen propagation constant at all
    (off by a factor of 4-32x, and the two drives' own independent fits of
    the SAME array disagree with each other by 1-2 orders of magnitude —
    see ``tests/test_coax_msl_transition.py``'s
    ``test_coax_msl_transition_first_fixture_diagnostic`` for the locked
    assertions and
    ``test_post_review_discriminant_msl_ladder_too_short_for_pencil_fit``
    for the pure-geometry version of the same check). Whether the
    junction's own physical reflection ALSO contributes is genuinely
    UNRESOLVED by this one fixture — a preflight advisory on the same
    fixture (MSL port too close to its own x-CPML face) names a third,
    also-unruled-out candidate mechanism — but the extraction-class
    explanation is the better-supported one and is what a future retry
    (a longer MSL probe ladder, not attempted in this PR) would target.
    Do not read the reciprocity/degeneracy numbers on this result as
    evidence about coax<->MSL launch physics in general.

    ATTEMPT 2 (PI-directed, R2's escape clause — attempt 1's own named
    defect authorized exactly one retry): lengthened the MSL probe ladder
    1.000mm -> 8.000mm and widened the MSL port's x-CPML clearance
    200um -> 1500um, keeping the junction geometry byte-identical to
    attempt 1 (asserted, see
    ``tests/test_coax_msl_transition.py::
    test_attempt2_junction_geometry_is_byte_identical_to_attempt1``).

    Verdict, per the RUN-LENGTH INVARIANCE TEST across two settling
    checkpoints (20000 -> 45000 steps; issue #585 adversarial review):
    the fitted ``gamma`` is the ONE quantity that is stable across both
    checkpoints and lands inside the predeclared [0.8, 1.3] band both
    times — CONFIRMED, but PROVISIONAL pending a fully settled run
    (neither checkpoint clears the -40 dB ring-down rule; see
    ``tests/test_coax_msl_transition.py``'s ``SETTLED_RUN_RECORD``, a
    predeclared-UNRUN follow-up targeting VESSL). Reciprocity (0.824 ->
    0.938), ``|S22|`` (0.451 -> 1.104), and max ``|S|`` (0.993 -> 1.104,
    crossing the passivity-guard hard limit) all FAIL that same
    invariance test — still evolving between checkpoints, in the WRONG
    direction — so these are **UNMEASURED at this settling**, not
    "refuted with cause identified" (an earlier revision of this
    docstring made exactly that overclaim; see below).

    RETRACTED (do not repeat — the third retracted attribution on this
    lane, after attempt 1's own "near-degenerate two-drive amplification"):
    a prior revision attributed the reciprocity miss to a coax/MSL
    drive-amplitude gap (~1.8e7-3.3e7x). This is mathematically
    impossible — per-drive (column) rescaling of the two-drive solve
    leaves ``s_params`` EXACTLY invariant by construction (verified
    numerically at this attempt's own gap value: deviation ~3e-16) — and
    the "amplitude ratio" invoked turned out to equal raw ``cond_a`` to 8
    significant figures, the exact quantity this docstring's own
    ``cond_a`` / ``cond_a_equilibrated`` split already says not to read
    as a degeneracy witness on this lane. The productive, SCALING-
    INVARIANT open question in its place: the MSL-driven column's power
    (``sum_j |s_params[j, 1, :]|**2``) is mostly far below 1 at both
    checkpoints (0.0018-0.204, rising to 0.0104-1.218) on a nominally
    lossless structure — where does that power go? Not answered by
    attempt 2; the next step is the settled VESSL run, not a third
    ladder/clearance change.

    DISCLOSURE (issue #585 final-verify, finding G1): the shared passivity
    guard both attempts rely on (``rfx/validation.py``'s ``check_passivity``
    block, ``strict_passivity=True`` path — see
    :func:`rfx.api._sparams._finalize_sparam_result`) checks only whether
    ``max column power`` EXCEEDS its upper limit; it has no lower-bound
    check at all, so a column power far BELOW 1 on a lossless structure —
    exactly the open question above — passes it silently. That one-sided
    guard is part of why the col_power finding went undetected across both
    attempts: nothing in the pipeline flags "too little" power, only "too
    much." This is a disclosure, not a fix — the guard is unchanged by
    this attempt.

    SETTLED RUN (VESSL 369367252283, 2026-08-06, ``n_steps=135000`` — see
    ``tests/test_coax_msl_transition.py``'s ``SETTLED_RUN_RECORD``, status
    ``RUN``): both drives clear the -40 dB ring-down rule for the first
    time on this lane (settling_db -45.94 / -44.17 dB). gamma-vs-beta is
    now **CONFIRMED**, not provisional — a THIRD checkpoint (after 20000
    and 45000 steps) still lands inside [0.8, 1.3] (ratio 1.148 / 0.859 /
    1.051). Passivity is **ATTRIBUTED**: settled max ``|S|`` = 0.9933 and
    the strict passivity guard raises nothing — the earlier
    passivity-guard trip (max ``|S|`` 1.104, column power 1.218, both >1
    at the 45000-step checkpoint) is now understood as a TRUNCATION
    artifact of an unsettled run, not a real violation. Reciprocity is
    now **MEASURED**, not merely disclosed — worst deviation 91.4% (pair
    (0, 1)) AT FULL SETTLING, explained by NONE of this lane's three
    retracted attributions (near-degenerate drives; the drive-amplitude
    gap, proven impossible; the MSL-ladder instrument-scoping limit,
    resolved by the gamma-vs-beta pass above, not by this number). Per
    this lane's own retraction history, that 91.4% is reported as a
    measurement, not adjudicated between a genuine loss/coupling
    asymmetry and a surviving instrument limitation. THE OPEN QUESTION is
    correspondingly sharpened, not answered: at full settling the
    MSL-driven column power is 0.00653 / 0.01098 / 0.79865 at 6/8/10 GHz —
    ~99.3% / ~98.9% / ~20.1% of incident power unaccounted for, dropping
    sharply toward 10 GHz (derived ``|S12|**2`` stays ~1e-7 throughout —
    essentially nothing transmits to the coax side; the retained power is
    almost entirely ``|S22|**2``). Two named, NOT-adjudicated candidates:
    (a) physical — the unmatched vertical launch radiates the MSL drive's
    power into the CPML absorber at low frequency (consistent with the
    frequency trend); (b) instrument — the MSL-side outgoing-wave
    extraction misses non-quasi-TEM content near the junction. A named
    discriminating check (not run): a closed-box (PEC-wall, no absorber)
    variant of this fixture — power reappearing in the port accounting
    would support (a); power still missing would support (b). SUPERSEDED
    as the chosen instrument: the issue #589 pre-declaration (2026-08-07
    comment) replaces the closed-box variant with face-resolved flux
    accounting on the settled open fixture (see ``extra_flux_monitors=``
    and the ``flux_monitors`` attribute below), with the deviation's
    reasoning recorded there.

    Attributes
    ----------
    s_params : (2, 2, n_freqs) complex
        ``s_params[j, i, :]`` = response at port *j* driving port *i*.
        Port order is ALWAYS ``("coax", "msl")`` — port 0 = coax, port 1 =
        MSL (see ``port_names``).
    freqs : (n_freqs,) float
    port_names : tuple of str
        Always ``("coax", "msl")``.
    reference_planes : (2,) float
        ``[junction_z_m, junction_x_m]`` — the coax port's own z reference
        plane and the MSL port's own x reference plane. Both are placed AT
        the physical launch discontinuity (see class docstring); they are
        NOT on a shared axis and must not be subtracted from one another.
    z0_ref : (2,) float
        ``[z0_coax_ohm, z0_msl_ohm]`` — the power-wave reference impedance
        used for each port: analytic coax TEM Z0
        (:func:`rfx.sources.coaxial_port.coaxial_tem_characteristic_impedance`)
        and analytic Hammerstad-Jensen microstrip Zc
        (:func:`rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff`).
        NOTE: the registered ``add_coaxial_port(impedance=...)`` /
        ``add_msl_port(impedance=...)`` values are NOT used here (they only
        size the feed resistor / termination and, for coax, the source
        amplitude calibration) — :meth:`_SparamMixin.compute_coax_msl_transition`
        warns when a registered impedance diverges from the analytic value
        actually used by more than 5%.
    cond_a : (n_freqs,) float
        Per-frequency condition number of the two-drive incident-wave
        matrix, RAW (same contract as :class:`CoaxialTwoPortResult`'s own
        ``cond_a``). On a mixed-family lane this is dominated by the two
        drives' unrelated source-amplitude SCALES (see the finding above)
        and is NOT a reliable geometric-degeneracy discriminant here —
        use ``cond_a_equilibrated`` for that.
    cond_a_equilibrated : (n_freqs,) float
        Condition number of the SAME incident-wave matrix after dividing
        each drive's own column by its own norm — invariant to per-drive
        amplitude scale, so it isolates genuine geometric near-parallelism
        between the two drives' incident waves. Does not affect
        ``s_params`` (column equilibration leaves ``S = B @ inv(A)``
        unchanged). Added in response to issue #581 review finding B2.
    recurrence_residual, fit_residual, gamma : (2, 2, n_freqs)
        Per (port array, drive, freq), same convention and meaning as
        :class:`CoaxialTwoPortResult` (0 = clean single-mode field; the MSL
        array's own numbers are the SAME diagnostics applied to the MSL
        probe ladder rather than a coax probe ladder). On the one committed
        fixture the MSL array's ``fit_residual`` crosses the predeclared
        "large = unreliable" line at two of six (array, drive, freq)
        entries, and its ``gamma`` does not track the analytic beta (see
        the finding above) — inspect these before trusting any MSL-side
        number from a new fixture.
    a_inc, b_out : (2, 2, n_freqs) complex
        The POWER-wave incident/outgoing amplitudes actually fed to the
        two-drive solve (after the ``sqrt(Z0)`` division), exposed for
        audit (issue #581 review finding B2) — e.g. to check each drive's
        own excitation actually reached a comparable amplitude before
        trusting ``cond_a``/``cond_a_equilibrated``.
    settling_db : (2,) float
        Ring-down settling witness per drive (worst end/peak field-energy
        ratio, dB; above -40 dB = truncation suspect — see repo ring-down
        convention). Since issue #662 that bar is ENFORCED, not just
        documented: a violating drive emits a ``UserWarning`` (never an
        exception) naming the drive and its measured value.
    status : str
        ``"experimental"`` always (this lane makes no pass/fail physics
        claim beyond what the calling test's own predeclared gate states;
        unlike :class:`CoaxialTwoPortResult` there is no committed accuracy
        battery to derive "passed"/"contaminated" from yet).
    """

    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    reference_planes: np.ndarray
    z0_ref: np.ndarray
    cond_a: np.ndarray
    cond_a_equilibrated: np.ndarray
    recurrence_residual: np.ndarray
    fit_residual: np.ndarray
    gamma: np.ndarray
    a_inc: np.ndarray
    b_out: np.ndarray
    settling_db: np.ndarray
    status: str = "experimental"
    flux_monitors: dict | None = None


__all__ = [
    "MATERIAL_LIBRARY",
    "AD_MemoryEstimate",
    "ADMemoryPlan",
    "ADMemoryComponent",
    "ADMemoryActionHint",
    "ADMemoryExplainabilityReport",
    "ADMemoryPreflightReport",
    "MeshIntelligenceReport",
    "ADCompiledMemoryCertificate",
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
    "CoaxialTwoPortResult",
    "_MSLPortEntry",
    "MSLSMatrixResult",
    "MixedSMatrixResult",
    "CoaxMSLTransitionResult",
]
