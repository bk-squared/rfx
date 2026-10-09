"""AD-memory planning methods for :class:`rfx.api.Simulation`."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import jax

from rfx.grid import C0
from rfx.materials.thin_conductor import has_f0_sheets
from rfx.api._spec import (
    AD_MemoryEstimate,
    ADMemoryPlan,
    ADMemoryComponent,
    ADMemoryActionHint,
    ADMemoryExplainabilityReport,
    ADMemoryPreflightReport,
    ADCompiledMemoryCertificate,
)
from rfx.api._compiled_memory import (
    _canonicalize_exact_scope,
    _environment_summary,
    _normalize_compiled_memory_analysis,
)
from rfx.api._validators import (
    _require_integral_param,
    _require_positive_finite_scalar,
    _positive_divisors,
    _validate_residual_context,
)


AD_MEMORY_FIT_SAFETY_FACTOR = 1.30


AD_MEMORY_PREFLIGHT_EVIDENCE_BOUNDARIES = (
    "static memory planning only",
    "trace-time JAX saved-residual explainability only when residual_diagnostic is present",
    "not a runtime peak-memory guarantee",
    "not XLA memory analysis",
    "not profiler evidence",
    "not a certificate",
    "not RF validation",
)


AD_COMPILED_MEMORY_CERTIFICATE_EVIDENCE_BOUNDARIES = (
    "bounded to one caller-supplied compiled executable",
    "uses JAX Compiled.memory_analysis() estimates",
    "not a universal guaranteed peak-memory predictor",
    "not profiler evidence",
    "not RF validation",
    "digests are audit identities, not executable correspondence proof",
)


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


class _AdMemoryMixin:
    # ---- AD memory estimation (issue #30 CHECK 4) ----

    def _ad_memory_static_accounting(self) -> dict[str, int]:
        """Return shared static byte accounting for AD memory artifacts.

        Issue #696: the cell count comes from the grid the SOLVE WILL
        ACTUALLY BUILD — ``_build_nonuniform_grid()`` when any mesh
        profile is set, ``_build_grid()`` otherwise — not from a private
        re-derivation of the shape. The re-derivation
        (``ceil(extent/dx) + 1 + 2*cpml_layers`` per axis) silently
        disagreed with both real grids wherever the shape depends on
        anything it did not model: per-face pads (a ``pec``/``pmc``/
        ``periodic`` face allocates 0 cells on that side) and 2-D mode
        (``nz == 1``). Measured on a 20x20mm graded-dz stackup, dx=0.2mm,
        cpml_layers=8: with PEC y faces the re-derivation says 0.780 M
        cells against the real NU grid's 0.674 M (+15.7 %); in
        ``mode="2d_tmz"`` it says 1.602 M against the real 0.014 M
        (117x). The estimate is what a user sizes a GPU with, so a grid
        it does not describe is worse than no estimate.

        ``grid_kind`` names which grid the numbers describe and travels
        with the estimate into its artifact — a uniform-lane ``dt`` has
        already been mistaken for the NU one here.
        """
        dx = self._dx or (C0 / self._freq_max / 20.0)
        is_nonuniform = any(
            p is not None
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        )
        grid_kind = "nonuniform" if is_nonuniform else "uniform"
        grid_source = "built"
        try:
            _grid = (self._build_nonuniform_grid() if is_nonuniform
                     else self._build_grid())
            nx, ny, nz = (int(v) for v in _grid.shape)
        except Exception:
            # Keep a number rather than raising out of a planning helper;
            # the fallback is the pre-#696 arithmetic and is LABELLED as
            # such so the artifact does not present it as the real grid.
            def _nx(extent: float, prof) -> int:
                if prof is not None:
                    return len(prof) + 1 + 2 * self._cpml_layers
                return int(math.ceil(extent / dx)) + 1 + 2 * self._cpml_layers

            nx = _nx(self._domain[0], self._dx_profile)
            ny = _nx(self._domain[1], self._dy_profile)
            nz = _nx(self._domain[2], self._dz_profile)
            grid_source = "estimated_from_domain"
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
        # Surface-impedance (f0) sheet operator (#677) — three boolean
        # tangential edge masks plus one float32 sigma_sheet array, all
        # full-grid, resident for the whole run. Counted here because
        # they are NOT part of the six material arrays above: since #677
        # the sheet touches neither pec_mask nor materials.sigma, so the
        # estimate for a lossy board was the estimate for the SAME board
        # without loss — which is exactly the model that then fit on the
        # same GPU (#696).
        sheet_bytes = 0
        if has_f0_sheets(getattr(self, "_thin_conductors", ())):
            sheet_bytes = cells * (3 * 1 + bytes_per_cell)
        forward_bytes = field_bytes + material_bytes + cpml_bytes + sheet_bytes

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
            "sheet_bytes": sheet_bytes,
            "forward_bytes": forward_bytes,
            "ntff_bytes": ntff_bytes,
            "grid_kind": grid_kind,
            "grid_source": grid_source,
        }


    def estimate_ad_memory(
        self,
        n_steps: int,
        *,
        available_memory_gb: float | None = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
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
        from issue #73.

        The segmented model (issue #277) counts BOTH terms of the
        rematerialized backward pass:

        * segment-boundary storage — carry + cotangent per active segment,
          ``2 * active_segments * field_bytes``;
        * the live-segment rematerialization tape — the backward pass
          replays one segment at a time, so that segment's per-step field
          tape (``n_steps // checkpoint_segments`` steps on the uniform
          path, ``checkpoint_every`` steps on the chunked path) is resident
          on top of the boundary storage. The non-uniform runner pads the
          trailing chunk with zero source to a full chunk, so the full
          chunk length is the correct live-tape size; both live terms are
          capped at the active (post-warmup) step count.

        The segmented estimate is therefore minimized when the two terms
        balance, near ``sqrt(2 * n_steps)`` steps per segment — NOT by
        pushing the segment count to either extreme. Use
        :meth:`plan_ad_memory` to pick a balanced knob for a budget.
        ``n_warmup`` must be an integer with
        ``0 <= n_warmup < n_steps`` and reduces reported reverse-mode tape
        time. The legacy
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
        field_bytes = accounting["field_bytes"]
        forward_bytes = accounting["forward_bytes"]
        ntff_bytes = accounting["ntff_bytes"]

        # Legacy "checkpointed" estimate: remat-recomputes internals of
        # step_fn. NOT valid for the NU path after issue #31 because the
        # scan carry itself is not rematerialised.
        ad_ckpt_bytes = 4 * forward_bytes + ntff_bytes
        active_steps = n_steps_i - n_warmup_i
        active_tape_field_bytes = field_bytes

        # Non-checkpointed AD: O(active_steps) full-grid field tape.
        # ``n_warmup`` reduces active reverse-mode time.
        ad_full_bytes = active_steps * active_tape_field_bytes + ntff_bytes + forward_bytes

        # Segmented scan paths. ``checkpoint_every`` is the non-uniform
        # scan-of-scan chunk length (issue #31); ``checkpoint_segments`` is
        # the uniform segmented-scan segment count (issue #73). Both store
        # carry + cotangent at segment boundaries, AND during the backward
        # pass the segment currently being differentiated is rematerialized
        # as a whole, so its per-step field tape is live at the same time
        # (issue #277; the runners document peak as O((K + s) · |carry|)).
        ad_seg_bytes: int | None = None
        segmented_active_segments: int | None = None
        segmented_live_tape_steps: int | None = None
        if checkpoint_segments_i is not None:
            segment_len = n_steps_i // checkpoint_segments_i
            first_active_segment = n_warmup_i // segment_len
            segmented_active_segments = checkpoint_segments_i - first_active_segment
            segmented_live_tape_steps = min(segment_len, active_steps)
            ad_seg_bytes = (
                (2 * segmented_active_segments + segmented_live_tape_steps)
                * active_tape_field_bytes
                + forward_bytes + ntff_bytes
            )
        elif checkpoint_every_i is not None:
            total_segments = math.ceil(n_steps_i / checkpoint_every_i)
            inactive_segments = n_warmup_i // checkpoint_every_i
            segmented_active_segments = total_segments - inactive_segments
            # The non-uniform runner pads the trailing chunk with zero
            # source up to a full ``checkpoint_every`` steps, so the live
            # rematerialized chunk is a full chunk (capped at the active
            # reverse-mode step count).
            segmented_live_tape_steps = min(checkpoint_every_i, active_steps)
            ad_seg_bytes = (
                (2 * segmented_active_segments + segmented_live_tape_steps)
                * active_tape_field_bytes
                + forward_bytes + ntff_bytes
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
            # Direction-aware advice (#277): with the live-segment tape in
            # the model, the segmented peak is minimized when the boundary
            # term (2 · active_segments) and the live term (steps per
            # segment) balance near sqrt(2 · n_steps). Point the user
            # toward the dominant term.
            live_dominates = (
                segmented_live_tape_steps is not None
                and segmented_active_segments is not None
                and segmented_live_tape_steps
                > 2 * segmented_active_segments
            )
            if checkpoint_segments_i is not None:
                if segmented_active_segments == 1:
                    action = "Reduce grid size, reduce n_steps, or use a more aggressive memory-reduction lane."
                elif live_dominates:
                    action = "Increase checkpoint_segments toward sqrt(n_steps) to shrink the live-segment tape, reduce grid size, or reduce n_steps."
                else:
                    action = "Reduce checkpoint_segments, reduce grid size, or reduce n_steps."
            elif ad_seg_bytes is not None:
                if segmented_active_segments == 1:
                    action = "Reduce grid size, reduce n_steps, or use a more aggressive memory-reduction lane."
                elif live_dominates:
                    action = "Reduce checkpoint_every toward sqrt(n_steps) to shrink the live-chunk tape, reduce grid size, or reduce n_steps."
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
            ad_segmented_active_segments=segmented_active_segments,
            grid_kind=accounting["grid_kind"],
            grid_source=accounting["grid_source"],
            grid_shape=(accounting["nx"], accounting["ny"], accounting["nz"]),
            sheet_gb=accounting["sheet_bytes"] * to_gb,
        )

    def explain_ad_memory(
        self,
        n_steps: int,
        *,
        available_memory_gb: float | None = None,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
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
        )

        accounting = self._ad_memory_static_accounting()
        field_bytes = accounting["field_bytes"]
        material_bytes = accounting["material_bytes"]
        cpml_bytes = accounting["cpml_bytes"]
        sheet_bytes = accounting["sheet_bytes"]
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
            # Live-segment rematerialization tape (#277): backward replays
            # one segment at a time, so that segment's per-step field tape
            # is resident on top of the boundary storage. Mirrors the
            # estimate_ad_memory accounting.
            if estimate.checkpoint_segments is not None:
                live_tape_steps = min(
                    n_steps_i // int(estimate.checkpoint_segments), active_steps
                )
            else:
                live_tape_steps = min(int(estimate.checkpoint_every), active_steps)
        else:
            live_tape_steps = None
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

        live_components: tuple[ADMemoryComponent, ...] = ()
        if live_tape_steps is not None:
            live_components = (
                _component(
                    "segmented_live_segment_tape",
                    live_tape_steps * field_bytes,
                    "reverse_ad_saved_state",
                    unit="live-segment-time-step-field-state",
                    count=int(live_tape_steps),
                    bytes_per_unit=field_bytes,
                    explanation=(
                        "The backward pass rematerializes one segment at a "
                        "time, so the live segment's per-step field tape is "
                        "resident on top of the segment-boundary storage."
                    ),
                ),
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
            *live_components,
            _component(
                "surface_impedance_sheet_state",
                sheet_bytes,
                "forward_working_set",
                unit="sheet-operator-array",
                count=None,
                bytes_per_unit=None,
                explanation=(
                    "Surface-impedance (surface_impedance_f0) sheet "
                    "operator state: three boolean tangential edge masks "
                    "plus sigma_sheet, all full-grid. Zero when no f0 "
                    "sheet is registered. Since #677 the sheet is in "
                    "neither pec_mask nor materials.sigma, so nothing "
                    "else in this accounting covers it (#696)."
                ),
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
                f"Selected strategy stores {tape_count} {tape_unit} unit(s) "
                f"plus one live rematerialized segment of {live_tape_steps} "
                f"time-step tape entries, instead of {active_steps} active "
                "time-step tape entries."
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
        ``n_warmup`` must be an integer with ``0 <= n_warmup < n_steps`` and
        reduces active reverse-mode time. Raw estimates must also fit after
        multiplying by ``safety_factor`` before fit flags are set, so
        near-boundary plans stay conservative.
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

            # With the live-segment term (#277) the least-memory candidate
            # is the divisor balancing the boundary and live tape terms,
            # NOT checkpoint_segments=1 (which is ~full-AD-sized). Report
            # the true least-memory divisor as the diagnostic candidate.
            least_segments: int | None = None
            least_estimate: AD_MemoryEstimate | None = None
            for segments in sorted(divisors):
                estimate = self.estimate_ad_memory(
                    n_steps_i,
                    available_memory_gb=available_memory_gb_f,
                    checkpoint_segments=segments,
                    n_warmup=n_warmup_i,
                )
                segmented_gb = estimate.ad_segmented_gb
                if segmented_gb is None:
                    continue
                if least_estimate is None or segmented_gb < least_estimate.ad_segmented_gb:
                    least_segments = segments
                    least_estimate = estimate
            least_fits = (
                least_estimate is not None
                and least_estimate.ad_segmented_gb * safety_factor_f
                <= target_memory_gb
            )
            if least_fits:
                recommendation = (
                    f"use checkpoint_segments={least_segments}: segmented AD estimate "
                    f"{_format_memory_with_safety(least_estimate.ad_segmented_gb, safety_factor_f)} fits within the "
                    f"{_format_memory_gb(target_memory_gb)} target"
                )
            else:
                recommendation = (
                    f"no checkpoint_segments divisor fits; the least-memory candidate "
                    f"checkpoint_segments={least_segments} is estimated at "
                    f"{_format_memory_with_safety(least_estimate.ad_segmented_gb, safety_factor_f)}, above the "
                    f"{_format_memory_gb(target_memory_gb)} target; reduce mesh size, reduce "
                    "n_steps, or use a more aggressive memory-reduction lane"
                )
            return ADMemoryPlan(
                n_steps=n_steps_i,
                available_memory_gb=available_memory_gb_f,
                target_fraction=target_fraction_f,
                target_memory_gb=target_memory_gb,
                checkpoint_every=None,
                checkpoint_segments=least_segments,
                checkpoint_mode="checkpoint_segments",
                fit_safety_factor=safety_factor_f,
                selected_estimate=least_estimate,
                full_ad_fits=False,
                segmented_fits=least_fits,
                recommendation=recommendation,
            )

        best_checkpoint: int | None = None
        best_estimate: AD_MemoryEstimate | None = None
        least_checkpoint: int | None = None
        least_estimate: AD_MemoryEstimate | None = None
        for checkpoint in range(1, n_steps_i + 1):
            estimate = self.estimate_ad_memory(
                n_steps_i,
                available_memory_gb=available_memory_gb_f,
                checkpoint_every=checkpoint,
                n_warmup=n_warmup_i,
            )
            segmented_gb = estimate.ad_segmented_gb
            if segmented_gb is not None and (
                least_estimate is None
                or segmented_gb < least_estimate.ad_segmented_gb
            ):
                least_checkpoint = checkpoint
                least_estimate = estimate
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

        # The full 1..n_steps sweep found no fitting chunk, so nothing
        # fits. With the live-chunk term (#277) the least-memory candidate
        # balances the boundary and live tape terms near
        # sqrt(2 * n_steps), NOT checkpoint_every=n_steps (which is
        # ~full-AD-sized) — report the tracked least-memory candidate.
        return ADMemoryPlan(
            n_steps=n_steps_i,
            available_memory_gb=available_memory_gb_f,
            target_fraction=target_fraction_f,
            target_memory_gb=target_memory_gb,
            checkpoint_every=least_checkpoint,
            checkpoint_segments=None,
            checkpoint_mode="checkpoint_every",
            fit_safety_factor=safety_factor_f,
            selected_estimate=least_estimate,
            full_ad_fits=False,
            segmented_fits=False,
            recommendation=(
                f"no checkpoint_every value fits; the least-memory candidate "
                f"checkpoint_every={least_checkpoint} is estimated at "
                f"{_format_memory_with_safety(least_estimate.ad_segmented_gb, safety_factor_f)}, above the "
                f"{_format_memory_gb(target_memory_gb)} target; reduce mesh size, reduce "
                "n_steps, or use a more aggressive memory-reduction lane"
            ),
        )

    def ad_memory_preflight(
        self,
        n_steps: int,
        available_memory_gb: float,
        *,
        target_fraction: float = 0.85,
        n_warmup: int = 0,
        safety_factor: float = AD_MEMORY_FIT_SAFETY_FACTOR,
        include_mesh_report: bool = True,
        check_ntff: bool = True,
        check_resolution: bool = True,
        residual_fun: Callable[..., Any] | None = None,
        residual_args: tuple[Any, ...] = (),
        residual_kwargs: Mapping[str, Any] | None = None,
        residual_top_n: int = 10,
        residual_workflow: str | None = None,
        residual_context: Mapping[str, object] | None = None,
    ) -> ADMemoryPreflightReport:
        """Return a one-call AD-memory preflight planning artifact.

        The report composes existing static memory planning, static
        explainability, optional mesh advisories, and optional trace-time JAX
        saved-residual diagnostics. It does not run FDTD and does not upgrade
        static planning evidence into runtime memory proof.
        """
        plan = self.plan_ad_memory(
            n_steps,
            available_memory_gb,
            target_fraction=target_fraction,
            n_warmup=n_warmup,
            safety_factor=safety_factor,
        )

        if plan.full_ad_fits:
            status = "full_ad_fits"
            selected_checkpoint_every = None
            selected_checkpoint_segments = None
            supported_checkpoint_mode = None
        else:
            selected_checkpoint_every = plan.checkpoint_every
            selected_checkpoint_segments = plan.checkpoint_segments
            supported_checkpoint_mode = plan.checkpoint_mode
            status = "checkpointing_fits" if plan.segmented_fits else "does_not_fit"

        explanation = self.explain_ad_memory(
            n_steps,
            available_memory_gb=available_memory_gb,
            checkpoint_every=selected_checkpoint_every,
            checkpoint_segments=selected_checkpoint_segments,
            n_warmup=n_warmup,
        )

        mesh_report = (
            self.mesh_intelligence_report(
                n_steps=n_steps,
                checkpoint_every=selected_checkpoint_every,
                checkpoint_segments=selected_checkpoint_segments,
                available_memory_gb=available_memory_gb,
                n_warmup=n_warmup,
                check_ntff=check_ntff,
                check_resolution=check_resolution,
            )
            if include_mesh_report
            else None
        )

        action_hints: list[ADMemoryActionHint] = []
        if plan.full_ad_fits:
            recommendation = "Full reverse-mode AD fits the conservative memory target; checkpointing is optional for memory."
            action_hints.append(
                ADMemoryActionHint(
                    code="full_ad_fits",
                    severity="info",
                    message="Full reverse-mode AD fits the configured conservative memory target.",
                    action="Run full reverse-mode AD or still choose checkpointing for extra margin.",
                )
            )
        elif plan.segmented_fits:
            if plan.checkpoint_mode == "checkpoint_every":
                code = "use_checkpoint_every"
                action = (
                    f"Wire checkpoint_every={plan.checkpoint_every} into the supported non-uniform runner."
                )
            else:
                code = "use_checkpoint_segments"
                action = (
                    f"Wire checkpoint_segments={plan.checkpoint_segments} into the supported uniform runner."
                )
            recommendation = plan.recommendation
            action_hints.append(
                ADMemoryActionHint(
                    code=code,
                    severity="warning",
                    message="Supported segmented checkpointing fits the configured conservative memory target.",
                    action=action,
                    checkpoint_mode=plan.checkpoint_mode,
                    checkpoint_every=plan.checkpoint_every,
                    checkpoint_segments=plan.checkpoint_segments,
                )
            )
        else:
            recommendation = (
                f"{plan.recommendation}; do not launch the returned checkpoint "
                "candidate as a fit. It is diagnostic-only."
            )
            action_hints.append(
                ADMemoryActionHint(
                    code="memory_budget_unfit",
                    severity="blocker",
                    message="Neither full reverse-mode AD nor the least-memory supported checkpoint candidate fits the configured conservative memory target.",
                    action=(
                        "Do not launch the returned checkpoint candidate as a fit; "
                        "reduce mesh size, reduce n_steps, increase memory, or choose a stronger memory-reduction lane."
                    ),
                    checkpoint_mode=plan.checkpoint_mode,
                    checkpoint_every=plan.checkpoint_every,
                    checkpoint_segments=plan.checkpoint_segments,
                    blocking=True,
                )
            )

        owned_residual_context = {
            "n_steps": int(plan.n_steps),
            "available_memory_gb": float(plan.available_memory_gb),
            "target_fraction": float(plan.target_fraction),
            "target_memory_gb": float(plan.target_memory_gb),
            "fit_safety_factor": float(plan.fit_safety_factor),
            "checkpoint_mode": supported_checkpoint_mode,
            "checkpoint_every": selected_checkpoint_every,
            "checkpoint_segments": selected_checkpoint_segments,
            "preflight_status": status,
        }
        # Validate whenever context is supplied (a malformed context must be
        # rejected early even without residual_fun), but warn if a valid
        # context is provided without a function, since it is then unused.
        merged_residual_context = (
            _validate_residual_context(
                residual_context,
                owned_fields=owned_residual_context,
            )
            if residual_fun is not None or residual_context is not None
            else None
        )
        if residual_context is not None and residual_fun is None:
            import warnings

            warnings.warn(
                "residual_context is ignored because residual_fun was not "
                "provided; pass residual_fun to produce a saved-residual "
                "diagnostic.",
                stacklevel=2,
            )
        residual_diagnostic = None
        if residual_fun is not None:
            from rfx.ad_diagnostics import diagnose_ad_saved_residuals
            artifact_snapshots: dict[str, object] = {
                "memory_plan": plan,
                "explainability": explanation,
            }
            if mesh_report is not None:
                artifact_snapshots["mesh_report"] = mesh_report
            residual_diagnostic = diagnose_ad_saved_residuals(
                residual_fun,
                *residual_args,
                top_n=residual_top_n,
                workflow=residual_workflow,
                context=merged_residual_context,
                artifacts=artifact_snapshots,
                **(dict(residual_kwargs) if residual_kwargs is not None else {}),
            )
            action_hints.append(
                ADMemoryActionHint(
                    code="inspect_saved_residuals",
                    severity="info",
                    message="JAX saved-residual trace context is attached for comparison with static memory artifacts.",
                    action="Compare top residuals and groups with the static explanation before changing checkpoint policy.",
                )
            )

        action_hints.append(
            ADMemoryActionHint(
                code="validate_physics_separately",
                severity="info",
                message="Memory preflight is separate from electromagnetic correctness checks.",
                action="Run convergence, reference, or observable checks before making physics claims.",
            )
        )

        return ADMemoryPreflightReport(
            n_steps=plan.n_steps,
            available_memory_gb=plan.available_memory_gb,
            target_fraction=plan.target_fraction,
            target_memory_gb=plan.target_memory_gb,
            fit_safety_factor=plan.fit_safety_factor,
            status=status,
            supported_checkpoint_mode=supported_checkpoint_mode,
            checkpoint_every=selected_checkpoint_every,
            checkpoint_segments=selected_checkpoint_segments,
            full_ad_fits=plan.full_ad_fits,
            checkpointing_fits=plan.segmented_fits,
            memory_plan=plan,
            explainability=explanation,
            mesh_report=mesh_report,
            residual_diagnostic=residual_diagnostic,
            action_hints=tuple(action_hints),
            evidence_boundaries=AD_MEMORY_PREFLIGHT_EVIDENCE_BOUNDARIES,
            recommendation=recommendation,
        )

    def ad_memory_compiled_certificate(
        self,
        compiled: object,
        *,
        n_steps: int,
        available_memory_gb: float,
        target_fraction: float = 0.85,
        checkpoint_every: int | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
        precision: str | None = None,
        input_signature: object | None = None,
        static_signature: object | None = None,
        compiled_object_id: str | None = None,
        runner_or_objective: str | None = None,
        preflight: ADMemoryPreflightReport | None = None,
        scope_context: Mapping[str, object] | None = None,
    ) -> ADCompiledMemoryCertificate:
        """Return a fail-closed compiler-memory certificate for one executable.

        This method accepts an already-compiled JAX object and reads only its
        ``memory_analysis()`` output. It does not compile callables, run FDTD,
        profile runtime memory, or convert static memory plans into guarantees.
        Green statuses are bounded to the supplied compiled object and complete
        exact-scope metadata.
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
        available_memory_gb_f = _require_positive_finite_scalar(
            "available_memory_gb",
            available_memory_gb,
        )
        target_fraction_f = _require_positive_finite_scalar(
            "target_fraction",
            target_fraction,
        )
        if target_fraction_f > 1.0:
            raise ValueError("target_fraction must be in the interval (0, 1]")

        # Reuse the existing AD-memory validation contract for counts and
        # checkpoint knob exclusivity/divisibility.
        self.estimate_ad_memory(
            n_steps_i,
            available_memory_gb=available_memory_gb_f,
            checkpoint_every=checkpoint_every_i,
            checkpoint_segments=checkpoint_segments_i,
            n_warmup=n_warmup_i,
        )

        target_memory_gb = available_memory_gb_f * target_fraction_f
        analysis = _normalize_compiled_memory_analysis(compiled)
        memory_analysis_fields = (
            tuple(analysis.fields)
            if analysis.status == "complete"
            else None
        )
        scope = _canonicalize_exact_scope(
            compiled=compiled,
            n_steps=n_steps_i,
            n_warmup=n_warmup_i,
            checkpoint_every=checkpoint_every_i,
            checkpoint_segments=checkpoint_segments_i,
            available_memory_gb=available_memory_gb_f,
            target_fraction=target_fraction_f,
            target_memory_gb=target_memory_gb,
            precision=precision,
            input_signature=input_signature,
            static_signature=static_signature,
            compiled_object_id=compiled_object_id,
            runner_or_objective=runner_or_objective,
            memory_analysis_fields=memory_analysis_fields,
            preflight=preflight,
            scope_context=scope_context,
        )

        utilization_ratio: float | None = None
        if scope.status != "complete":
            status = scope.status
            status_reason = scope.reason
        elif analysis.status != "complete":
            status = analysis.status
            status_reason = analysis.reason
        else:
            required_bytes = int(analysis.required_bytes)
            target_bytes = target_memory_gb * 1e9
            utilization_ratio = (
                required_bytes / target_bytes if target_bytes > 0 else None
            )
            if required_bytes <= target_bytes:
                status = "compiler_estimate_within_budget"
                status_reason = (
                    "compiled memory_analysis() required bytes fit within "
                    "available_memory_gb * target_fraction"
                )
            else:
                status = "compiler_estimate_exceeds_budget"
                status_reason = (
                    "compiled memory_analysis() required bytes exceed "
                    "available_memory_gb * target_fraction"
                )

        def _bytes(field: str) -> int | None:
            return analysis.fields.get(field) if analysis.status == "complete" else None

        def _gb(value: int | None) -> float | None:
            return None if value is None else value / 1e9

        temp_bytes = _bytes("temp_size_in_bytes")
        argument_bytes = _bytes("argument_size_in_bytes")
        output_bytes = _bytes("output_size_in_bytes")
        alias_bytes = _bytes("alias_size_in_bytes")
        required_bytes_out = (
            int(analysis.required_bytes)
            if analysis.status == "complete"
            else None
        )
        env = _environment_summary()
        exact_scope = scope.exact_scope
        jax_version = (
            str(exact_scope.get("jax_version"))
            if exact_scope is not None and "jax_version" in exact_scope
            else str(env["jax_version"])
        )
        recommendations = [
            "Treat this as a bounded certificate for the exact compiled object and declared scope only.",
            "Digests are audit identities only; they do not prove source-to-executable correspondence.",
        ]
        if status == "compiler_estimate_within_budget":
            util_txt = (
                f" (compiler estimate is {utilization_ratio * 100:.1f}% of the "
                "target budget)"
                if utilization_ratio is not None
                else ""
            )
            recommendations.insert(
                0,
                "Compiler memory analysis fits the target budget for this exact "
                f"scope{util_txt}. This is a JAX compiler estimate; it excludes "
                "allocator fragmentation and runtime scratch, so a fit at high "
                "utilization can still OOM at runtime.",
            )
        elif status == "compiler_estimate_exceeds_budget":
            recommendations.insert(
                0,
                "Compiler memory analysis exceeds the target budget; reduce scope, checkpoint more aggressively, or increase memory.",
            )
        elif status == "scope_incomplete":
            recommendations.insert(
                0,
                "Provide complete JSON-safe exact-scope metadata before relying on compiler memory evidence.",
            )
        elif status == "scope_mismatch":
            recommendations.insert(
                0,
                "Resolve contradictions between declared scope, preflight metadata, and compiled-object introspection.",
            )
        elif status == "analysis_unavailable":
            recommendations.insert(
                0,
                "JAX did not provide memory_analysis() evidence for this compiled object/backend.",
            )
        else:
            recommendations.insert(
                0,
                "JAX memory_analysis() output is incomplete or invalid for certificate use.",
            )

        return ADCompiledMemoryCertificate(
            status=status,
            status_reason=status_reason,
            available_memory_gb=available_memory_gb_f,
            target_fraction=target_fraction_f,
            target_memory_gb=target_memory_gb,
            compiler_reported_required_bytes=required_bytes_out,
            compiler_reported_required_gb=_gb(required_bytes_out),
            temp_size_in_bytes=temp_bytes,
            argument_size_in_bytes=argument_bytes,
            output_size_in_bytes=output_bytes,
            alias_size_in_bytes=alias_bytes,
            temp_gb=_gb(temp_bytes),
            argument_gb=_gb(argument_bytes),
            output_gb=_gb(output_bytes),
            alias_gb=_gb(alias_bytes),
            exact_scope=exact_scope,
            scope_status=scope.status,
            scope_status_reason=scope.reason,
            scope_digest=scope.scope_digest,
            config_digest=scope.config_digest,
            environment_digest=scope.environment_digest,
            memory_analysis_status=analysis.status,
            memory_analysis_status_reason=analysis.reason,
            jax_version=jax_version,
            evidence_boundaries=AD_COMPILED_MEMORY_CERTIFICATE_EVIDENCE_BOUNDARIES,
            recommendations=tuple(recommendations),
            source_preflight=scope.source_preflight,
        )
