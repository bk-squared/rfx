"""AD-memory planning methods for :class:`rfx.api.Simulation`."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import jax

from rfx.stepping.remat import active_levels, checkpoint_lengths, checkpoint_memory, plan_checkpoint_lengths, validate_cotangents

from rfx.api._ad_memory_model import (
    tape_accounting, segment_counts, tape_warning, validate_counts, face_record_resident_bytes,
)
from rfx.backend import backend_class
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


AD_MEMORY_FIT_SAFETY_FACTOR = 1.15


def _checkpoint_param(name, value):
    if isinstance(value, tuple):
        if name == "checkpoint_segments":
            raise ValueError("AD memory checkpoint_segments must be an int; use checkpoint_every for levels")
        return checkpoint_lengths(value)
    return _require_integral_param(name, value)


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
        _grid = None
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
            "grid": _grid,
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
        checkpoint_every: int | tuple[int, ...] | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
        design_box: object | None = None,
        design: str = "permittivity",
        checkpoint: bool = True,
        distributed: bool = False,
        cotangents: int = 1,
    ) -> "AD_MemoryEstimate":
        """Estimate saved carry, per-step tape, and calibrated AD working set.

        ``design_box`` accepts forward's corners/region forms. The window
        comes from the same realized-grid resolver. Whole-grid tape saves
        the real scan carry except declared linear accumulators; box tape
        saves six permittivity or three occupancy window arrays. A rematted, unsegmented box uses a labelled
        carry-plus-window bound. The fixed term is calibrated on GPU with
        JAX 0.10.2, rather than a bound for CPU compiler temporaries.
        ``cotangents`` outputs pulled back at once add 7.5 carries each beyond
        the first: one boundary carry plus 6.5 carries of working arrays.
        This is calibrated on one graded whole-grid scene (GPU, JAX 0.10.2).
        Adjoint fields and backward working arrays remain whole-grid with a
        design box; the same workspace there is an unmeasured estimate.

        Segments save one carry per active boundary and one live segment,
        independently of the per-step checkpoint flag.
        Balance K_active*carry_bytes against s*tape_step_bytes. Warmup
        reduces the reverse-mode time. The legacy ad_checkpointed_gb
        heuristic remains for compatibility; use ad_full_gb or the requested
        ad_segmented_gb for planning. These are estimates, not certificates.
        """
        cotangents = validate_cotangents(cotangents)
        n_steps_i = _require_integral_param("n_steps", n_steps)
        n_warmup_i = _require_integral_param("n_warmup", n_warmup)
        checkpoint_every_i = (
            None
            if checkpoint_every is None
            else _checkpoint_param("checkpoint_every", checkpoint_every)
        )
        checkpoint_segments_i = (
            None
            if checkpoint_segments is None
            else _checkpoint_param("checkpoint_segments", checkpoint_segments)
        )
        avail_gb = (
            None
            if available_memory_gb is None
            else _require_positive_finite_scalar(
                "available_memory_gb", available_memory_gb
            )
        )

        validate_counts(n_steps_i, n_warmup_i, checkpoint_every_i, checkpoint_segments_i)
        accounting = self._ad_memory_static_accounting()
        segmented = checkpoint_segments_i is not None or checkpoint_every_i is not None
        if isinstance(checkpoint_every_i, tuple):
            segmented = bool(active_levels(n_steps_i - n_warmup_i, checkpoint_every_i))
        model = tape_accounting(
            self, accounting, design_box=design_box, design=design,
            checkpoint=checkpoint, distributed=distributed, n_steps=n_steps_i,
            checkpoint_every=checkpoint_every_i, checkpoint_segments=checkpoint_segments_i, n_warmup=n_warmup_i,
            segmented=segmented)
        forward_bytes = accounting["forward_bytes"]
        ntff_bytes = accounting["ntff_bytes"]
        replay_table_bytes = model["replay_table_bytes"]
        batched_bytes = (cotangents - 1) * (model["working_bytes"] + model["carry_bytes"])
        ad_ckpt_bytes = 4 * forward_bytes + ntff_bytes + batched_bytes  # legacy heuristic
        active_steps = n_steps_i - n_warmup_i
        fixed_bytes = model["c0_bytes"] + accounting["sheet_bytes"] + model["resident_bytes"]
        ad_full_bytes = fixed_bytes + active_steps * model["tape_step_bytes"] + ntff_bytes + replay_table_bytes + batched_bytes
        segmented_active_segments, segmented_live_tape_steps = segment_counts(
            n_steps_i, n_warmup_i, checkpoint_every_i, checkpoint_segments_i)
        ad_seg_bytes = None
        if segmented_active_segments is not None:
            saved = segmented_active_segments * model["carry_bytes"]
            ad_seg_bytes = (fixed_bytes + saved
                            + segmented_live_tape_steps * model["tape_step_bytes"] + ntff_bytes + replay_table_bytes + batched_bytes)

        # Device memory detection (best effort). CPU has no device budget.
        memory_limit_unknown = False
        if avail_gb is None:
            try:
                memory_limit_unknown = backend_class().device_memory == "memory_stats_if_available"
                devs = jax.local_devices()
                for d in devs:
                    policy = backend_class(d.platform).device_memory
                    if policy not in ("memory_stats", "memory_stats_if_available"):
                        continue
                    if policy == "memory_stats":
                        stats = d.memory_stats() if hasattr(d, "memory_stats") else None
                        if stats and "bytes_limit" in stats:
                            avail_gb = stats["bytes_limit"] / 1e9
                            break
                        continue
                    memory_limit_unknown |= policy == "memory_stats_if_available"
                    try:
                        stats = d.memory_stats() if hasattr(d, "memory_stats") else None
                        limit = stats.get("bytes_limit") if stats else None
                        if (not isinstance(limit, bool) and limit is not None
                                and math.isfinite(limit) and limit > 0):
                            avail_gb = limit / 1e9
                            break
                    except Exception:
                        continue
            except Exception:
                avail_gb = None

        to_gb = 1.0 / 1e9
        # Pick the most realistic estimate for the warning: segmented if
        # requested, otherwise full-AD (since the legacy "checkpointed"
        # number is unreliable — see class docstring).
        primary_bytes = ad_seg_bytes if ad_seg_bytes is not None else ad_full_bytes
        warning = tape_warning(model, checkpoint=checkpoint,
                               segments=segmented_active_segments,
                               live=segmented_live_tape_steps, active=active_steps)
        if isinstance(checkpoint_every_i, tuple):
            memory = checkpoint_memory(active_steps, checkpoint_every_i,
                boundary_bytes=model["carry_bytes"], step_bytes=model["tape_step_bytes"],
                static_bytes=model["c0_bytes"], cotangents=cotangents, working_bytes=model["working_bytes"])
            warning = " ".join(filter(None, (warning,
                f"Checkpoint lengths {memory.lengths}: {memory.levels} levels, "
                f"{memory.forward_passes} forward passes (estimate).")))
        if memory_limit_unknown and avail_gb is None:
            warning = " ".join(filter(None, (
                "The device memory limit cannot be read on this backend; "
                "AD memory fit cannot be judged. Supply available_memory_gb explicitly.",
                warning)))
        if avail_gb is not None and primary_bytes * to_gb > avail_gb * 0.85:
            budget_warning = (
                f"AD memory estimate {_format_memory_gb(primary_bytes * to_gb)} "
                f"exceeds 85% of {_format_memory_gb(avail_gb)} available VRAM. "
                "Use plan_ad_memory(), reduce grid size, or reduce n_steps.")
            warning = " ".join(filter(None, (budget_warning, warning)))
        return AD_MemoryEstimate(
            saved=tuple(model["saved"]), excluded_linear=tuple(model["excluded_linear"]),
            carry_warnings=tuple(model["carry_warnings"]),
            modelled=model["modelled"], unmodelled_features=tuple(model["unmodelled_features"]),
            checkpoint=checkpoint, design=design,
            cotangents=cotangents,
            carry_bytes=model["carry_bytes"], working_bytes=model["working_bytes"],
            tape_step_bytes=model["tape_step_bytes"], tape_window_cells=model["tape_window_cells"],
            design_box_bounds=model["design_box_bounds"], tape_is_bound=model["tape_is_bound"],
            ad_fixed_gb=fixed_bytes * to_gb,
            forward_gb=forward_bytes * to_gb,
            ad_checkpointed_gb=ad_ckpt_bytes * to_gb,
            ad_full_gb=ad_full_bytes * to_gb,
            ntff_dft_gb=ntff_bytes * to_gb,
            replay_table_bytes=replay_table_bytes,
            replay_table_bytes_by_kind=model["replay_table_bytes_by_kind"],
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
        self, n_steps: int, *, available_memory_gb: float | None = None,
        checkpoint_every: int | tuple[int, ...] | None = None, checkpoint_segments: int | None = None,
        n_warmup: int = 0, design_box: object | None = None,
        design: str = "permittivity", checkpoint: bool = True, distributed: bool = False,
        cotangents: int = 1,
    ) -> ADMemoryExplainabilityReport:
        """Decompose the same calibrated working set and saved tape as the estimate."""
        estimate = self.estimate_ad_memory(
            n_steps, available_memory_gb=available_memory_gb,
            checkpoint_every=checkpoint_every, checkpoint_segments=checkpoint_segments,
            n_warmup=n_warmup, design_box=design_box, design=design,
            checkpoint=checkpoint, distributed=distributed, cotangents=cotangents)
        accounting = self._ad_memory_static_accounting()
        segmented = estimate.ad_segmented_gb is not None
        selected = estimate.ad_segmented_gb if segmented else estimate.ad_full_gb
        strategy = ("segmented_checkpoint_segments" if checkpoint_segments is not None
                    else "segmented_checkpoint_every" if checkpoint_every is not None
                    else "full_reverse_ad_static_tape")
        parts = []

        def component(name, size, kind, *, unit=None, count=None, per=None, explanation=""):
            parts.append(ADMemoryComponent(
                name=name, memory_gb=size / 1e9, share_of_selected=size / 1e9 / selected,
                kind=kind, unit=unit, count=count,
                bytes_per_unit_gb=per / 1e9 if per is not None else None,
                explanation=explanation))

        array = accounting["cells"] * 4
        for name, key, count in (
                ("field_state", "field_bytes", 6),
                ("material_auxiliary_state", "material_bytes", 6),
                ("cpml_auxiliary_state", "cpml_bytes", None),
                ("surface_impedance_sheet_state", "sheet_bytes", None)):
            component(name, accounting[key], "forward_working_set", count=count,
                      per=array if count else None,
                      explanation="Forward working-set component within the calibrated fixed AD term.")
        resident = face_record_resident_bytes(self, accounting["grid"], int(n_steps),
            checkpoint=checkpoint, checkpoint_segments=checkpoint_segments,
            checkpoint_every=checkpoint_every, n_warmup=n_warmup)
        if resident:
            component("closed_box_face_record", resident, "forward_working_set",
                      explanation="One resident record of the incident operands read by the faces.")
        component("calibrated_ad_workspace",
                  round(estimate.ad_fixed_gb * 1e9) - accounting["forward_bytes"] - resident,
                  "calibrated_working_set",
                  explanation="GPU, jax 0.10.2 calibrated fixed term minus the listed forward working set.")
        if cotangents > 1:
            component("additional_cotangent_working_set", (cotangents - 1) * (estimate.working_bytes + estimate.carry_bytes),
                      "reverse_ad_working_set", count=cotangents - 1, per=estimate.working_bytes + estimate.carry_bytes,
                      explanation="7.5 carries per extra cotangent: one boundary plus 6.5 working; "
                      "calibrated on one graded whole-grid scene (GPU, JAX 0.10.2); design boxes unmeasured.")
        count, live = segment_counts(int(n_steps), int(n_warmup), checkpoint_every, checkpoint_segments)
        if segmented:
            component("segmented_boundary_field_tape", count * estimate.carry_bytes,
                      "reverse_ad_saved_state", count=count, per=estimate.carry_bytes,
                      unit="field-and-CPML-carry",
                      explanation="One saved field/CPML carry per active segment boundary.")
        else:
            component("full_reverse_field_tape", estimate.ad_active_steps * estimate.tape_step_bytes,
                      "reverse_ad_saved_state", count=estimate.ad_active_steps,
                      per=estimate.tape_step_bytes, unit="active-time-step-tape",
                      explanation="Formulation-specific per-step tape for the full active time record.")
        if segmented:
            component("segmented_live_segment_tape", live * estimate.tape_step_bytes,
                      "reverse_ad_saved_state", count=live, per=estimate.tape_step_bytes,
                      unit="live-segment-time-step-tape", explanation="One live rematerialized segment.")
        component("ntff_dft_state", accounting["ntff_bytes"], "monitor_state",
                  explanation="NTFF DFT state counted separately from the calibrated fixed term.")
        for kind, size in estimate.replay_table_bytes_by_kind:
            component(f"{kind}_replay_tables", size, "reverse_ad_saved_state",
                      explanation=f"Post-scan {kind} replay weights retained by the pullback, padded to replay blocks.")
        dominant = max(parts, key=lambda c: c.memory_gb).name
        recommendations = [f"Dominant AD memory contributor is {dominant}.",
                           "Treat ad_checkpointed_gb as a legacy heuristic; use the selected memory field for planning."]
        if estimate.warning:
            recommendations.append(estimate.warning)
        return ADMemoryExplainabilityReport(
            n_steps=int(n_steps), strategy=strategy, selected_memory_gb=selected,
            selected_memory_field="ad_segmented_gb" if segmented else "ad_full_gb",
            estimate=estimate, components=tuple(parts), dominant_component=dominant,
            recommendations=tuple(recommendations))

    def plan_ad_memory(
        self,
        n_steps: int,
        available_memory_gb: float,
        *,
        target_fraction: float = 0.85,
        n_warmup: int = 0,
        safety_factor: float = AD_MEMORY_FIT_SAFETY_FACTOR,
        design_box: object | None = None,
        design: str = "permittivity",
        checkpoint: bool = True,
        distributed: bool = False,
        max_levels: int | None = None,
        checkpoint_every: int | tuple[int, ...] | None = None,
        cotangents: int = 1,
    ) -> ADMemoryPlan:
        """Compare full AD with segmented candidates using checkpoint=True.

        The selected plan returns the checkpoint flag it assumes. Uniform
        candidates are divisors, in ascending K order, retaining the smallest
        fitting K; graded candidates retain the smallest fitting chunk.
        On graded single-device meshes, ``max_levels`` enables the nested
        length planner, preferring fewer levels; None keeps the original
        one-level search. ``checkpoint_every`` evaluates an explicit int or
        tuple candidate. The selected estimate reports active lengths,
        levels and forward passes without changing result fields.
        A box also tries the window-only, unsegmented checkpoint=False lane.
        A non-fitting result is diagnostic, never permission to launch.
        plan_ad_memory promises XLA's requirement (memory_analysis:
        argument + output + temp − alias) times SF, within
        target_fraction × budget. Allocator overhead and rounding are what
        target_fraction (0.85) is for. Under a non-default growth-mode setup
        the allocator was seen to round the temp buffer to the next power
        of two (G2: 3 of 6 rows, pool_bytes non-zero although the platform
        allocator was requested). SF is not inflated to cover rounding.
        Reading the real carry costs about 90 bytes of host memory per cell
        and a few seconds, measured on 4.3e7 cells.
        The fixed term is GPU-calibrated, so a CPU plan is not a fit guarantee;
        plan_ad_memory traces twice and ad_memory_preflight up to four times;
        closed-box plans additionally trace each candidate's representation.
        Nested planning also traces the segmented carry and selected lengths.
        """
        cotangents = validate_cotangents(cotangents)
        available = _require_positive_finite_scalar("available_memory_gb", available_memory_gb)
        graded = any(p is not None for p in (self._dx_profile, self._dy_profile, self._dz_profile))
        if max_levels is not None:
            max_levels = _require_integral_param("max_levels", max_levels)
            if max_levels < 1:
                raise ValueError("max_levels must be >= 1")
            if not graded or distributed:
                raise ValueError("max_levels requires the graded single-device path; uniform/distributed are unsupported")
        fraction = _require_positive_finite_scalar("target_fraction", target_fraction)
        safety = _require_positive_finite_scalar("safety_factor", safety_factor)
        if fraction > 1:
            raise ValueError("target_fraction must be in the interval (0, 1]")
        if safety < 1:
            raise ValueError("safety_factor must be >= 1")
        options = dict(design_box=design_box, design=design, distributed=distributed, cotangents=cotangents)
        full = self.estimate_ad_memory(n_steps, available_memory_gb=available,
                                      n_warmup=n_warmup, checkpoint=checkpoint, **options)
        window_full = self.estimate_ad_memory(n_steps, available_memory_gb=available,
                                             n_warmup=n_warmup, checkpoint=False, **options)
        estimate_warning = " ".join(window_full.carry_warnings) or None
        if window_full.ad_full_gb < full.ad_full_gb:
            full = window_full
        target = available * fraction
        n = _require_integral_param("n_steps", n_steps)
        warmup = _require_integral_param("n_warmup", n_warmup)

        def result(estimate, *, full_fits=False, segmented_fits=False):
            mode = ("checkpoint_every" if estimate.checkpoint_every is not None
                    else "checkpoint_segments" if estimate.checkpoint_segments is not None else None)
            memory = estimate.ad_segmented_gb if mode else estimate.ad_full_gb
            if not estimate.modelled:
                advice = "No fitting plan: unmodelled carry components " + ", ".join(estimate.unmodelled_features)
            elif full_fits:
                advice = f"full reverse-mode AD with checkpoint={estimate.checkpoint} fits; segmented checkpointing is optional for memory"
            elif segmented_fits:
                knob = estimate.checkpoint_every if mode == "checkpoint_every" else estimate.checkpoint_segments
                advice = f"use {mode}={knob}, checkpoint=True; balance K_active*carry_bytes = s*tape_step_bytes"
            else:
                knob = estimate.checkpoint_every if mode == "checkpoint_every" else estimate.checkpoint_segments
                advice = (f"no {mode} candidate fits; the least-memory candidate {mode}={knob}, "
                          "checkpoint=True is diagnostic only; reduce mesh size, reduce n_steps, or increase memory")
            return ADMemoryPlan(
                n_steps=n, available_memory_gb=available, target_fraction=fraction,
                target_memory_gb=target, checkpoint_every=estimate.checkpoint_every,
                checkpoint_segments=estimate.checkpoint_segments, checkpoint_mode=mode,
                checkpoint=estimate.checkpoint, fit_safety_factor=safety,
                selected_estimate=estimate, full_ad_fits=full_fits, segmented_fits=segmented_fits,
                recommendation=f"{advice}; estimate {_format_memory_with_safety(memory, safety)}, target {_format_memory_gb(target)}"
                + (f"; {estimate.warning}" if isinstance(estimate.checkpoint_every, tuple) else ""))

        if checkpoint_every is not None:
            estimate = self.estimate_ad_memory(n, available_memory_gb=available,
                n_warmup=warmup, checkpoint=True, checkpoint_every=checkpoint_every, **options)
            return result(estimate, segmented_fits=estimate.modelled and estimate.ad_segmented_gb * safety <= target)
        if not full.modelled:
            return result(full)
        if full.ad_full_gb * safety <= target:
            return result(full, full_fits=True)
        if max_levels is not None:
            accounting = self._ad_memory_static_accounting()
            model = tape_accounting(self, accounting, design_box=design_box, design=design,
                checkpoint=True, segmented=True, distributed=distributed, n_steps=n,
                checkpoint_every=1, n_warmup=warmup)
            extra = accounting["sheet_bytes"] + accounting["ntff_bytes"] + model["resident_bytes"] + model["replay_table_bytes"]
            plan = plan_checkpoint_lengths(n - warmup, boundary_bytes=model["carry_bytes"],
                step_bytes=model["tape_step_bytes"], static_bytes=model["c0_bytes"],
                cotangents=cotangents, working_bytes=model["working_bytes"],
                budget_bytes=target * 1e9 / safety - extra, max_levels=max_levels)
            estimate = self.estimate_ad_memory(n, available_memory_gb=available,
                n_warmup=warmup, checkpoint=True, checkpoint_every=plan.lengths, **options)
            peak = estimate.ad_segmented_gb if plan.lengths else estimate.ad_full_gb
            return result(estimate, segmented_fits=bool(plan.lengths) and peak * safety <= target)
        candidates = range(1, n + 1) if graded else _positive_divisors(n)
        least = None
        for knob in sorted(candidates):
            every, segments = (knob, None) if graded else (None, knob)
            count, live = segment_counts(n, warmup, every, segments)
            memory = (window_full.ad_fixed_gb + window_full.ntff_dft_gb
                      + window_full.replay_table_bytes / 1e9
                      + (count * window_full.carry_bytes + live * window_full.tape_step_bytes
                         + (cotangents - 1) * (window_full.working_bytes + window_full.carry_bytes)) / 1e9)
            estimate = window_full._replace(
                checkpoint=True, checkpoint_every=every, checkpoint_segments=segments,
                ad_segmented_active_segments=count, ad_segmented_gb=memory, warning=estimate_warning)
            if self._tfsf is not None and self._tfsf.closed_box:
                # Each candidate can choose a different representation and carry.
                estimate = self.estimate_ad_memory(n, available_memory_gb=available,
                    n_warmup=warmup, checkpoint=True, checkpoint_every=every,
                    checkpoint_segments=segments, **options)
                memory = estimate.ad_segmented_gb
            if least is None or memory < least.ad_segmented_gb:
                least = estimate
            if memory * safety <= target:
                return result(estimate, segmented_fits=True)
        return result(least)

    def ad_memory_preflight(
        self,
        n_steps: int,
        available_memory_gb: float,
        *,
        target_fraction: float = 0.85,
        n_warmup: int = 0,
        design_box: object | None = None,
        design: str = "permittivity",
        checkpoint: bool = True,
        distributed: bool = False,
        cotangents: int = 1,
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
        options = dict(design_box=design_box, design=design, checkpoint=checkpoint,
                       distributed=distributed, cotangents=cotangents)
        plan = self.plan_ad_memory(
            n_steps,
            available_memory_gb,
            target_fraction=target_fraction,
            n_warmup=n_warmup,
            safety_factor=safety_factor, **options,
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
            **dict(options, checkpoint=plan.checkpoint),
        )

        from rfx.mesh_planner import mesh_intelligence_report
        mesh_report = (
            mesh_intelligence_report(self,
                n_steps=n_steps,
                checkpoint_every=selected_checkpoint_every,
                checkpoint_segments=selected_checkpoint_segments,
                available_memory_gb=available_memory_gb,
                n_warmup=n_warmup,
                check_ntff=check_ntff,
                check_resolution=check_resolution,
                ad_memory_options=dict(options, checkpoint=plan.checkpoint),
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
                    f"Wire checkpoint=True, checkpoint_every={plan.checkpoint_every} into the supported non-uniform runner."
                )
            else:
                code = "use_checkpoint_segments"
                action = (
                    f"Wire checkpoint=True, checkpoint_segments={plan.checkpoint_segments} into the supported uniform runner."
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
            "checkpoint": plan.checkpoint,
            "design": design,
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
        checkpoint_every: int | tuple[int, ...] | None = None,
        checkpoint_segments: int | None = None,
        n_warmup: int = 0,
        design_box: object | None = None,
        design: str = "permittivity",
        checkpoint: bool = True,
        distributed: bool = False,
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
            else _checkpoint_param("checkpoint_every", checkpoint_every)
        )
        checkpoint_segments_i = (
            None
            if checkpoint_segments is None
            else _checkpoint_param("checkpoint_segments", checkpoint_segments)
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

        # A certificate inspects an existing executable; validation must not
        # trace forward or depend on a usable backend before fail-closed scope.
        validate_counts(n_steps_i, n_warmup_i, checkpoint_every_i, checkpoint_segments_i)
        if design not in ("permittivity", "occupancy"):
            raise ValueError("design must be 'permittivity' or 'occupancy'")
        if not isinstance(checkpoint, bool):
            raise ValueError("checkpoint must be a bool")
        box_bounds = None
        if design_box is not None:
            from rfx.model.design_box import design_box_bounds
            grid = self._build_realized_grid()
            box_bounds, _ = design_box_bounds(self._design_box_bounds_from_corners(grid, design_box), grid)

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
            ad_memory_scope={"checkpoint": checkpoint, "design": design,
                             "design_box_bounds": box_bounds, "distributed": distributed},
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
