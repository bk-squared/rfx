"""Leaf AD memory estimate record."""
from __future__ import annotations

import json
from typing import NamedTuple


class ADMemoryUnavailable(ValueError):
    """AD memory cannot be estimated for this precision or forward trace."""


class AD_MemoryEstimate(NamedTuple):
    """Reverse-mode tape estimate, calibrated on GPU with JAX 0.10.2.

    Sizes ending in ``_gb`` are decimal GB; ``_bytes`` fields are integers.
    Segments save one scan carry per active boundary
    plus a live segment, independently of the per-step checkpoint flag.
    The fixed AD working set is calibrated by
    formulation; it is distinct from the forward-only working set.
    ``cotangents`` outputs pulled back at once add 7.5 carries each beyond the
    first: one boundary carry plus 6.5 carries of working arrays, calibrated
    on one graded whole-grid scene (GPU, JAX 0.10.2). The same whole-grid
    adjoint workspace is estimated for design boxes; boxes are unmeasured.
    """
    forward_gb: float
    ad_checkpointed_gb: float
    ad_full_gb: float
    ntff_dft_gb: float
    available_gb: float | None
    warning: str | None
    ad_segmented_gb: float | None = None
    checkpoint_every: int | tuple[int, ...] | None = None
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
    saved: tuple[dict, ...] = ()
    excluded_linear: tuple[dict, ...] = ()
    modelled: bool = True
    unmodelled_features: tuple[str, ...] = ()
    checkpoint: bool = True
    design: str = "permittivity"
    carry_bytes: int | None = None
    tape_step_bytes: int | None = None
    tape_window_cells: int | None = None
    design_box_bounds: tuple[int, ...] | None = None
    ad_fixed_gb: float | None = None
    tape_is_bound: bool = False
    calibration: str = "calibrated (GPU, jax 0.10.2)"
    carry_warnings: tuple[str, ...] = ()
    cotangents: int = 1
    #: Post-scan replay weights, separate from scan carry and NTFF state.
    replay_table_bytes: int = 0
    #: Realized measurement kind and its summed retained replay weight bytes.
    replay_table_bytes_by_kind: tuple[tuple[str, int], ...] = ()
    #: Per-extra-cotangent working arrays, excluding the boundary carry.
    working_bytes: int = 0
    @property
    def evidence_class(self) -> str:
        """Evidence class label serialized with this static estimate."""
        return "static_estimate"


    def to_dict(self) -> dict[str, float | int | None | str]:
        """Return a stable JSON-serializable AD memory artifact."""
        return {
            "saved": list(self.saved),
            "excluded_linear": list(self.excluded_linear),
            "modelled": self.modelled,
            "unmodelled_features": list(self.unmodelled_features),
            "checkpoint": self.checkpoint,
            "design": self.design,
            "carry_bytes": self.carry_bytes,
            "working_bytes": self.working_bytes,
            "tape_step_bytes": self.tape_step_bytes,
            "tape_window_cells": self.tape_window_cells,
            "design_box_bounds": list(self.design_box_bounds) if self.design_box_bounds is not None else None,
            "ad_fixed_gb": self.ad_fixed_gb,
            "tape_is_bound": self.tape_is_bound,
            "calibration": self.calibration,
            "cotangents": self.cotangents,
            "evidence_class": self.evidence_class,
            "forward_gb": float(self.forward_gb),
            "ad_checkpointed_gb": float(self.ad_checkpointed_gb),
            "ad_full_gb": float(self.ad_full_gb),
            "ntff_dft_gb": float(self.ntff_dft_gb),
            "available_gb": (
                None if self.available_gb is None else float(self.available_gb)
            ),
            "warning": self.warning,
            "carry_warnings": list(self.carry_warnings),
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
