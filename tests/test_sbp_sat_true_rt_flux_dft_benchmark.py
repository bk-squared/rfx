"""Private flux/DFT benchmark evidence for SBP-SAT true R/T.

This file intentionally exercises a private benchmark-only path.  Public
``add_dft_plane_probe`` and ``add_flux_monitor`` remain hard-failing when
``Simulation`` uses SBP-SAT refinement, and the public ``Result`` surface is
not widened by the helper.
"""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import inspect
import json
from typing import NamedTuple
import warnings

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core.dft_utils import dft_window_weight
from rfx.probes.probes import flux_spectrum
from rfx.runners.subgridded import (
    _BenchmarkFluxPlaneRequest,
    _PrivateAnalyticSheetSourceRequest,
    _PrivateTFSFIncidentRequest,
    _build_benchmark_flux_plane_specs,
    _build_private_analytic_sheet_source_specs,
    _build_private_tfsf_incident_specs,
    run_private_tfsf_reference_flux,
    run_subgridded_benchmark_flux,
)
from rfx.sources.sources import CustomWaveform
from rfx.subgridding.jit_runner import (
    _BenchmarkFluxPlaneResult,
    _BenchmarkFluxPlaneSpec,
    _PrivateAnalyticSheetSourceSpec,
    _PrivateTFSFIncidentSpec,
    _accumulate_benchmark_flux_plane,
    _apply_private_tfsf_incident_e,
    _apply_private_tfsf_incident_h,
    _benchmark_flux_spectrum,
    _inject_private_analytic_sheet_source,
)
from rfx.subgridding.sbp_sat_3d import SubgridState3D


_STRICT_PLACEMENT = "fine-owned strict-interior"
_NO_GO_REASON = (
    "private analytic-sheet bounded-CPML fixture did not satisfy the "
    "incident-normalized fixture-quality gates required for public true R/T "
    "promotion"
)
_NEXT_PREREQUISITE = (
    "open a separate private TFSF-style incident-field fixture plan or private "
    "normalization-repair plan before reconsidering public true R/T, DFT, "
    "flux, port, or S-parameter promotion"
)
_TFSF_NEXT_PREREQUISITE = (
    "open a private solver/interface-floor investigation before low-level hook "
    "experiments or any public true R/T, DFT, flux, TFSF, port, or S-parameter "
    "promotion"
)
_INTERFACE_FLOOR_NEXT_PREREQUISITE = (
    "open a private SBP-SAT interface energy-transfer repair plan before "
    "low-level hook experiments or any public true R/T, DFT, flux, TFSF, port, "
    "or S-parameter promotion"
)
_PRIVATE_REPAIR_NO_MATERIAL_NEXT_PREREQUISITE = (
    "open a private SBP-SAT energy-transfer theory/design review before "
    "low-level hook experiments or any public true R/T, DFT, flux, TFSF, port, "
    "or S-parameter promotion"
)
_PRIVATE_REPAIR_ACCEPTED_NEXT_PREREQUISITE = (
    "open a private fixture-quality recovery plan using the accepted private "
    "energy-transfer candidate before any public promotion"
)
_PRIVATE_REPAIR_REPAIRED_NEXT_PREREQUISITE = (
    "open a private slab R/T scoring gate plan; keep public promotion deferred "
    "until a separate public-support ralplan approves it"
)
_PRIVATE_REPAIR_EVIDENCE_ARTIFACT_TEMPLATE = (
    ".omx/reports/sbp-sat-private-interface-energy-transfer-repair-{timestamp}.md"
)
_PRIVATE_TRUE_RT_SLOW_COMMAND = (
    "pytest -q tests/test_sbp_sat_true_rt_flux_dft_benchmark.py -m 'gpu and slow' -s"
)
_TFSF_NO_GO_REASON = (
    "private TFSF-style incident fixture has a same-contract reference, but "
    "vacuum/reference fixture-quality gates remain inconclusive; slab R/T "
    "scoring is intentionally skipped"
)
_NORMALIZATION_FLOOR = 1e-30
_NONFLOOR_FACTOR = 1e12
_MIN_CLAIMS_BEARING_BINS = 2
_TRANSVERSE_MAGNITUDE_CV_MAX = 0.01
_TRANSVERSE_PHASE_SPREAD_DEG_MAX = 1.0
_VACUUM_MAGNITUDE_ERROR_MAX = 0.02
_VACUUM_PHASE_ERROR_DEG_MAX = 2.0
_DOMINANT_IMPROVEMENT_MIN = 0.50
_PAIRED_IMPROVEMENT_MIN = 0.25
_NEW_BLOCKER_REGRESSION_MAX = 0.25

_ALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES = frozenset(
    {
        "measurement_or_reference_helper",
        "source_waveform_or_stagger",
        "plane_placement_or_phase_center",
        "finite_aperture_source_edge",
        "fixture_interface_geometry",
    }
)
_DISALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES = frozenset(
    {
        "sbp_sat_interface_floor",
        "private_hook_order_or_stagger",
        "unresolved_after_ladder",
    }
)
_INTERFACE_FLOOR_SUBCLASSES = frozenset(
    {
        "fixture_interface_geometry",
        "fixture_interface_geometry_candidate",
        "coarse_fine_energy_transfer_mismatch",
        "cpml_interface_proximity_artifact",
        "solver_interface_floor_persistent",
        "hook_contingency_direct_invariant",
        "unresolved_interface_floor",
    }
)
_PRIVATE_REPAIR_TAU_CANDIDATES = (0.25, 0.5, 0.75, 1.0)
_PRIVATE_REPAIR_STATUSES = frozenset(
    {
        "accepted_private_candidate",
        "repaired_private_floor",
        "no_material_repair",
    }
)
_FRONT_BACK_RATIO_FORMULA = "signed_back / max(abs(signed_front), floor)"
_FRONT_BACK_RATIO_ERROR_FORMULA = (
    "abs(subgrid_ratio - uniform_ratio) / max(abs(uniform_ratio), floor)"
)


class _GuardFixture:
    freqs = np.array([2.0e9], dtype=np.float64)


@dataclass(frozen=True)
class _FluxFixtureConfig:
    """Private benchmark fixture geometry with derived fine-grid metadata."""

    name: str
    fixture_key: str
    freq_max: float = 6.0e9
    source_freq: float = 2.0e9
    source_bandwidth: float = 0.8
    domain: tuple[float, float, float] = (0.04, 0.04, 0.09)
    cpml_layers: int = 4
    uniform_dx: float = 1.0e-3
    coarse_dx: float = 2.0e-3
    ratio: int = 2
    tau: float = 0.5
    n_steps: int = 700
    refinement_x_range: tuple[float, float] = (0.012, 0.028)
    refinement_y_range: tuple[float, float] = (0.012, 0.028)
    refinement_z_range: tuple[float, float] = (0.016, 0.074)
    sheet_coordinate: float = 0.024
    sheet_component: str = "ey"
    sheet_amplitude: float = 1.0e8
    sheet_phase_rad: float = 0.0
    sheet_x_span: tuple[float, float] = (0.013, 0.027)
    sheet_y_span: tuple[float, float] = (0.013, 0.027)
    front_plane: float = 0.028
    back_plane: float = 0.064
    aperture_center: tuple[float, float] = (0.020, 0.020)
    aperture_diameter: float = 0.014
    slab_lo: tuple[float, float, float] = (0.012, 0.012, 0.046)
    slab_hi: tuple[float, float, float] = (0.028, 0.028, 0.054)
    slab_eps_r: float = 2.25
    scored_freqs_tuple: tuple[float, ...] = (1.5e9, 2.0e9, 2.5e9)

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError("fixture ratio must be positive")
        if self.tau <= 0:
            raise ValueError("fixture tau must be positive")
        if not np.isclose(self.dx_f, self.uniform_dx):
            raise ValueError("fixture fine dx must match uniform fine reference dx")
        for label, span in {
            "refinement_x_range": self.refinement_x_range,
            "refinement_y_range": self.refinement_y_range,
            "refinement_z_range": self.refinement_z_range,
            "sheet_x_span": self.sheet_x_span,
            "sheet_y_span": self.sheet_y_span,
        }.items():
            if not span[0] < span[1]:
                raise ValueError(f"{label} must be increasing")
        if not (
            self.refinement_z_range[0]
            < self.sheet_coordinate
            < self.front_plane
            < self.slab_lo[2]
            < self.slab_hi[2]
            < self.back_plane
            < self.refinement_z_range[1]
        ):
            raise ValueError("fixture z ordering must be sheet < front < slab < back")
        for axis, (ref_span, sheet_span, slab_lo, slab_hi) in enumerate(
            (
                (
                    self.refinement_x_range,
                    self.sheet_x_span,
                    self.slab_lo[0],
                    self.slab_hi[0],
                ),
                (
                    self.refinement_y_range,
                    self.sheet_y_span,
                    self.slab_lo[1],
                    self.slab_hi[1],
                ),
            )
        ):
            if not (
                ref_span[0] < sheet_span[0] < sheet_span[1] < ref_span[1]
                and ref_span[0] <= slab_lo < slab_hi <= ref_span[1]
            ):
                raise ValueError(f"fixture tangential geometry invalid on axis {axis}")
        self._validate_refinement_alignment()
        self._validate_absorber_guard()

    def _validate_refinement_alignment(self) -> None:
        for span in (
            self.refinement_x_range,
            self.refinement_y_range,
            self.refinement_z_range,
        ):
            cells = (span[1] - span[0]) / self.coarse_dx
            if not np.isclose(cells, round(cells), atol=1e-9):
                raise ValueError("refinement ranges must align to coarse cells")

    def _validate_absorber_guard(self) -> None:
        guard = (self.cpml_layers + 1) * self.coarse_dx
        for axis, (span, domain_size) in enumerate(
            zip(
                (
                    self.refinement_x_range,
                    self.refinement_y_range,
                    self.refinement_z_range,
                ),
                self.domain,
                strict=True,
            )
        ):
            if span[0] < guard - 1e-12 or span[1] > domain_size - guard + 1e-12:
                raise ValueError(
                    f"fixture refinement must stay outside CPML guard on axis {axis}"
                )

    @property
    def refinement(self) -> dict[str, object]:
        return {
            "x_range": self.refinement_x_range,
            "y_range": self.refinement_y_range,
            "z_range": self.refinement_z_range,
            "ratio": self.ratio,
            "tau": self.tau,
        }

    @property
    def dx_f(self) -> float:
        return self.coarse_dx / self.ratio

    @property
    def shape_f(self) -> tuple[int, int, int]:
        # Mirrors the runner's coarse-cell inclusive refinement lowering.
        return tuple(
            int(round((hi - lo) / self.coarse_dx) + 1) * self.ratio
            for lo, hi in (
                self.refinement_x_range,
                self.refinement_y_range,
                self.refinement_z_range,
            )
        )

    @property
    def offsets(self) -> tuple[float, float, float]:
        return (
            self.refinement_x_range[0],
            self.refinement_y_range[0],
            self.refinement_z_range[0],
        )

    @property
    def source(self) -> tuple[float, float, float]:
        return (
            self.aperture_center[0],
            self.aperture_center[1],
            self.sheet_coordinate,
        )

    @property
    def source_component(self) -> str:
        return self.sheet_component

    @property
    def aperture_size(self) -> tuple[float, float]:
        return (self.aperture_diameter, self.aperture_diameter)

    @property
    def scored_freqs(self) -> np.ndarray:
        return np.array(self.scored_freqs_tuple, dtype=np.float64)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fixture": self.fixture_key,
            "domain": list(self.domain),
            "cpml_layers": self.cpml_layers,
            "coarse_dx": self.coarse_dx,
            "uniform_dx": self.uniform_dx,
            "ratio": self.ratio,
            "tau": self.tau,
            "n_steps": self.n_steps,
            "refinement": self.refinement,
            "shape_f": list(self.shape_f),
            "offsets": list(self.offsets),
            "sheet_coordinate": self.sheet_coordinate,
            "sheet_span": [list(self.sheet_x_span), list(self.sheet_y_span)],
            "front_plane": self.front_plane,
            "back_plane": self.back_plane,
            "aperture_center": list(self.aperture_center),
            "aperture_size": list(self.aperture_size),
            "slab_lo": list(self.slab_lo),
            "slab_hi": list(self.slab_hi),
        }


_FluxFixture = _FluxFixtureConfig(
    name="current_bounded",
    fixture_key="bounded_cpml_private_analytic_sheet_flux_plane_vacuum_slab",
)

_BoundaryExpandedFluxFixture = _FluxFixtureConfig(
    name="boundary_expanded",
    fixture_key="boundary_expanded_private_analytic_sheet_flux_plane_vacuum_slab",
    domain=(0.048, 0.048, 0.105),
    refinement_x_range=(0.010, 0.038),
    refinement_y_range=(0.010, 0.038),
    refinement_z_range=(0.014, 0.090),
    sheet_coordinate=0.026,
    sheet_x_span=(0.014, 0.034),
    sheet_y_span=(0.014, 0.034),
    front_plane=0.036,
    back_plane=0.078,
    aperture_center=(0.024, 0.024),
    aperture_diameter=0.020,
    slab_lo=(0.014, 0.014, 0.056),
    slab_hi=(0.034, 0.034, 0.066),
)

_RECOVERY_SWEEP_FIXTURES = (_FluxFixture, _BoundaryExpandedFluxFixture)

_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_STATUS = (
    "measurement_contract_or_interface_floor_persists"
)
_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_NEXT_PREREQUISITE = (
    "private measurement-contract/interface-floor redesign after helper recovery failed "
    "ralplan"
)
_PRIVATE_TIME_CENTERED_HELPER_C3_ROLLBACK_METRICS = {
    "usable_bins": 3,
    "transverse_magnitude_cv": 0.4518431429723011,
    "transverse_phase_spread_deg": 144.39582467090602,
    "vacuum_relative_magnitude_error": 0.8516395504247894,
    "vacuum_phase_error_deg": 20.354022359965228,
}
_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_LADDER = (
    {
        "candidate_id": "C0_current_helper_original_fixture",
        "candidate_type": "baseline_original_fixture",
        "parameters": {
            "fixture": "boundary_expanded",
            "source_kind": "private_tfsf",
            "plane_shift_cells": 0,
            "aperture_size": 0.020,
            "helper_relaxation": 0.02,
        },
        "solver_touch": False,
        "can_claim_original_fixture_recovery": True,
    },
    {
        "candidate_id": "C1_center_core_measurement_control",
        "candidate_type": "measurement_control",
        "parameters": {
            "fixture": "boundary_expanded",
            "source_kind": "private_tfsf",
            "plane_shift_cells": 0,
            "aperture_size": 0.010,
            "helper_relaxation": 0.02,
        },
        "solver_touch": False,
        "can_claim_original_fixture_recovery": False,
    },
    {
        "candidate_id": "C2_one_cell_downstream_plane_control",
        "candidate_type": "measurement_control",
        "parameters": {
            "fixture": "boundary_expanded",
            "source_kind": "private_tfsf",
            "plane_shift_cells": 1,
            "aperture_size": 0.020,
            "helper_relaxation": 0.02,
        },
        "solver_touch": False,
        "can_claim_original_fixture_recovery": False,
    },
    {
        "candidate_id": "C3_helper_relaxation_0p05_original_fixture",
        "candidate_type": "single_solver_local_candidate",
        "parameters": {
            "fixture": "boundary_expanded",
            "source_kind": "private_tfsf",
            "plane_shift_cells": 0,
            "aperture_size": 0.020,
            "helper_relaxation": 0.05,
            "only_allowed_solver_edit": (
                "_TIME_CENTERED_HELPER_RELAXATION = 0.02 -> 0.05"
            ),
        },
        "solver_touch": True,
        "can_claim_original_fixture_recovery": True,
    },
)


def _guard_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary="pec",
        dx=2e-3,
    )
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    sim.add_source(position=(0.020, 0.020, 0.020), component="ez")
    sim.add_probe(position=(0.020, 0.020, 0.022), component="ez")
    return sim


def _guard_reference_sim() -> Simulation:
    sim = Simulation(
        freq_max=5e9,
        domain=(0.04, 0.04, 0.04),
        boundary="pec",
        dx=1e-3,
    )
    sim.add_probe(position=(0.020, 0.020, 0.022), component="ex")
    return sim


def _private_guard_plane() -> _BenchmarkFluxPlaneRequest:
    return _BenchmarkFluxPlaneRequest(
        name="private_guard",
        axis="z",
        coordinate=0.020,
        freqs=_GuardFixture.freqs,
        size=(0.006, 0.006),
        center=(0.020, 0.020),
    )


def _private_guard_sheet() -> _PrivateAnalyticSheetSourceRequest:
    return _PrivateAnalyticSheetSourceRequest(
        name="private_guard_sheet",
        axis="z",
        coordinate=0.020,
        component="ey",
        propagation_sign=1,
        amplitude=1.0,
        f0_hz=2.0e9,
        bandwidth=0.8,
        phase_rad=0.0,
        x_span=(0.017, 0.023),
        y_span=(0.017, 0.023),
    )


def _private_guard_tfsf_incident() -> _PrivateTFSFIncidentRequest:
    return _PrivateTFSFIncidentRequest(
        name="private_guard_tfsf_incident",
        axis="z",
        coordinate=0.020,
        electric_component="ex",
        magnetic_component="hy",
        propagation_sign=1,
        amplitude=1.0,
        f0_hz=2.0e9,
        bandwidth=0.8,
        phase_rad=0.0,
        x_span=(0.017, 0.023),
        y_span=(0.017, 0.023),
    )


def test_public_dft_plane_probe_still_hard_fails_with_subgrid():
    sim = _guard_sim()
    sim.add_dft_plane_probe(
        axis="z",
        coordinate=0.020,
        component="ez",
        freqs=_GuardFixture.freqs,
    )

    with pytest.raises(ValueError, match="does not support DFT plane probes"):
        sim.run(n_steps=4)


def test_public_flux_monitor_still_hard_fails_with_subgrid():
    sim = _guard_sim()
    sim.add_flux_monitor(
        axis="z",
        coordinate=0.020,
        freqs=_GuardFixture.freqs,
        size=(0.006, 0.006),
        center=(0.020, 0.020),
    )

    with pytest.raises(ValueError, match="does not support flux monitors"):
        sim.run(n_steps=4)


def test_private_benchmark_run_does_not_populate_public_dft_or_flux_results():
    run = run_subgridded_benchmark_flux(
        _guard_sim(),
        n_steps=4,
        planes=(_private_guard_plane(),),
        sheet_sources=(_private_guard_sheet(),),
    )

    assert run.result.dft_planes is None
    assert run.result.flux_monitors is None
    assert len(run.benchmark_flux_planes) == 1
    assert run.benchmark_flux_planes[0].name == "private_guard"


def test_private_tfsf_benchmark_run_does_not_populate_public_dft_or_flux_results():
    run = run_subgridded_benchmark_flux(
        _guard_sim(),
        n_steps=4,
        planes=(_private_guard_plane(),),
        private_tfsf_incidents=(_private_guard_tfsf_incident(),),
    )

    assert run.result.dft_planes is None
    assert run.result.flux_monitors is None
    assert len(run.benchmark_flux_planes) == 1
    assert run.benchmark_flux_planes[0].name == "private_guard"


def test_private_tfsf_reference_run_does_not_populate_public_dft_or_flux_results():
    sim = _guard_reference_sim()
    assert sim._refinement is None

    run = run_private_tfsf_reference_flux(
        sim,
        n_steps=4,
        planes=(_private_guard_plane(),),
        private_tfsf_incidents=(_private_guard_tfsf_incident(),),
    )

    assert sim._refinement is None
    assert run.result.dft_planes is None
    assert run.result.flux_monitors is None
    assert len(run.benchmark_flux_planes) == 1
    assert run.benchmark_flux_planes[0].name == "private_guard"


def test_private_tfsf_reference_run_rejects_public_flux_requests():
    sim = _guard_reference_sim()
    sim.add_flux_monitor(
        axis="z",
        coordinate=0.020,
        freqs=_GuardFixture.freqs,
        size=(0.006, 0.006),
        center=(0.020, 0.020),
    )

    with pytest.raises(ValueError, match="rejects public observables"):
        run_private_tfsf_reference_flux(
            sim,
            n_steps=4,
            planes=(_private_guard_plane(),),
            private_tfsf_incidents=(_private_guard_tfsf_incident(),),
        )


def test_private_tfsf_reference_helper_uses_pre_cpml_private_slots():
    source = inspect.getsource(run_private_tfsf_reference_flux)

    h_update = source.index("state = update_h(")
    h_private = source.index("state = _apply_private_h_all(state, step_idx)")
    h_cpml = source.index("apply_cpml_h(")
    e_update = source.index("state = update_e(")
    e_private = source.index("state = _apply_private_e_all(state, step_idx)")
    e_cpml = source.index("apply_cpml_e(")

    assert h_update < h_private < h_cpml < e_update < e_private < e_cpml
    assert "run_uniform(" not in source
    assert "sim.run(" not in source


def _plane_specs(
    *planes: _BenchmarkFluxPlaneRequest,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_BenchmarkFluxPlaneSpec, ...]:
    return _build_benchmark_flux_plane_specs(
        planes,
        shape_f=fixture.shape_f,
        offsets=fixture.offsets,
        dx_f=fixture.dx_f,
        n_steps=fixture.n_steps,
    )


def _benchmark_sheet_source(
    *,
    fixture: _FluxFixtureConfig = _FluxFixture,
    coordinate: float | None = None,
    axis: str = "z",
    component: str | None = None,
    propagation_sign: int = 1,
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
) -> _PrivateAnalyticSheetSourceRequest:
    return _PrivateAnalyticSheetSourceRequest(
        name="private_sheet",
        axis=axis,
        coordinate=fixture.sheet_coordinate if coordinate is None else coordinate,
        component=fixture.sheet_component if component is None else component,
        propagation_sign=propagation_sign,
        amplitude=fixture.sheet_amplitude,
        f0_hz=fixture.source_freq,
        bandwidth=fixture.source_bandwidth,
        phase_rad=fixture.sheet_phase_rad,
        x_span=fixture.sheet_x_span if x_span is None else x_span,
        y_span=fixture.sheet_y_span if y_span is None else y_span,
    )


def _sheet_specs(
    *sources: _PrivateAnalyticSheetSourceRequest,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_PrivateAnalyticSheetSourceSpec, ...]:
    return _build_private_analytic_sheet_source_specs(
        sources,
        shape_f=fixture.shape_f,
        offsets=fixture.offsets,
        dx_f=fixture.dx_f,
        dt=1.0e-12,
        n_steps=fixture.n_steps,
    )


def _benchmark_tfsf_incident(
    *,
    fixture: _FluxFixtureConfig = _FluxFixture,
    coordinate: float | None = None,
    axis: str = "z",
    electric_component: str = "ex",
    magnetic_component: str = "hy",
    propagation_sign: int = 1,
    x_span: tuple[float, float] | None = None,
    y_span: tuple[float, float] | None = None,
) -> _PrivateTFSFIncidentRequest:
    return _PrivateTFSFIncidentRequest(
        name="private_tfsf_incident",
        axis=axis,
        coordinate=fixture.sheet_coordinate if coordinate is None else coordinate,
        electric_component=electric_component,
        magnetic_component=magnetic_component,
        propagation_sign=propagation_sign,
        amplitude=fixture.sheet_amplitude,
        f0_hz=fixture.source_freq,
        bandwidth=fixture.source_bandwidth,
        phase_rad=fixture.sheet_phase_rad,
        x_span=fixture.sheet_x_span if x_span is None else x_span,
        y_span=fixture.sheet_y_span if y_span is None else y_span,
    )


def _tfsf_specs(
    *incidents: _PrivateTFSFIncidentRequest,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_PrivateTFSFIncidentSpec, ...]:
    return _build_private_tfsf_incident_specs(
        incidents,
        shape_f=fixture.shape_f,
        offsets=fixture.offsets,
        dx_f=fixture.dx_f,
        dt=1.0e-12,
        n_steps=fixture.n_steps,
    )


def _benchmark_plane(
    name: str,
    *,
    coordinate: float,
    fixture: _FluxFixtureConfig = _FluxFixture,
    size: tuple[float, float] | None = None,
    center: tuple[float, float] | None = None,
) -> _BenchmarkFluxPlaneRequest:
    return _BenchmarkFluxPlaneRequest(
        name=name,
        axis="z",
        coordinate=coordinate,
        freqs=fixture.scored_freqs,
        size=fixture.aperture_size if size is None else size,
        center=fixture.aperture_center if center is None else center,
    )


def test_private_plane_accepts_strict_interior_fine_owned_planes():
    front, back = _plane_specs(
        _benchmark_plane("front", coordinate=_FluxFixture.front_plane),
        _benchmark_plane("back", coordinate=_FluxFixture.back_plane),
    )

    assert front.index == 12
    assert back.index == 48
    assert front.lo1 == front.lo2 == 1
    assert front.hi1 == front.hi2 == 15


def test_private_sheet_source_accepts_strict_interior_full_span():
    (sheet,) = _sheet_specs(_benchmark_sheet_source())

    assert sheet.axis == 2
    assert sheet.index == 8
    assert sheet.component == _FluxFixture.sheet_component
    assert sheet.propagation_sign == 1
    assert sheet.lo1 == sheet.lo2 == 1
    assert sheet.hi1 == sheet.hi2 == 15
    assert sheet.source_values.shape == (_FluxFixture.n_steps,)


def test_boundary_expanded_fixture_derives_strict_fine_grid_metadata():
    fixture = _BoundaryExpandedFluxFixture
    front, back = _plane_specs(
        _benchmark_plane("front", coordinate=fixture.front_plane, fixture=fixture),
        _benchmark_plane("back", coordinate=fixture.back_plane, fixture=fixture),
        fixture=fixture,
    )
    (sheet,) = _sheet_specs(
        _benchmark_sheet_source(fixture=fixture),
        fixture=fixture,
    )

    assert fixture.shape_f == (30, 30, 78)
    assert fixture.offsets == (0.010, 0.010, 0.014)
    assert front.index == 22
    assert back.index == 64
    assert front.lo1 == front.lo2 == 4
    assert front.hi1 == front.hi2 == 24
    assert sheet.index == 12
    assert sheet.lo1 == sheet.lo2 == 4
    assert sheet.hi1 == sheet.hi2 == 24
    assert sheet.source_values.shape == (fixture.n_steps,)


def test_private_tfsf_incident_accepts_strict_interior_ex_hy_pair():
    (incident,) = _tfsf_specs(_benchmark_tfsf_incident())

    assert incident.axis == 2
    assert incident.index == 8
    assert incident.electric_component == "ex"
    assert incident.magnetic_component == "hy"
    assert incident.propagation_sign == 1
    assert incident.lo1 == incident.lo2 == 1
    assert incident.hi1 == incident.hi2 == 15
    assert incident.electric_values.shape == (_FluxFixture.n_steps,)
    assert incident.magnetic_values.shape == (_FluxFixture.n_steps,)

    electric = np.asarray(incident.electric_values)
    magnetic = np.asarray(incident.magnetic_values)
    nonzero = np.abs(electric) > 1.0e-12 * np.max(np.abs(electric))
    eta = electric[nonzero] / magnetic[nonzero]
    assert np.all(np.isfinite(eta))
    assert 300.0 < float(np.mean(eta)) < 500.0


@pytest.mark.parametrize(
    "source",
    [
        _benchmark_sheet_source(axis="x"),
        _benchmark_sheet_source(component="ez"),
        _benchmark_sheet_source(propagation_sign=-1),
        _benchmark_sheet_source(x_span=(0.011, 0.027)),
    ],
)
def test_private_sheet_source_rejects_public_or_edge_touching_shapes(
    source: _PrivateAnalyticSheetSourceRequest,
):
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _sheet_specs(source)


@pytest.mark.parametrize(
    "incident",
    [
        _benchmark_tfsf_incident(axis="x"),
        _benchmark_tfsf_incident(electric_component="ey"),
        _benchmark_tfsf_incident(magnetic_component="hx"),
        _benchmark_tfsf_incident(propagation_sign=-1),
        _benchmark_tfsf_incident(x_span=(0.011, 0.027)),
    ],
)
def test_private_tfsf_incident_rejects_public_or_edge_touching_shapes(
    incident: _PrivateTFSFIncidentRequest,
):
    with pytest.raises(ValueError, match="TFSF-style incident fields"):
        _tfsf_specs(incident)


def test_private_sheet_source_injection_adds_selected_tangential_field_only():
    shape = (5, 6, 7)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    state = SubgridState3D(
        ex_c=zeros,
        ey_c=zeros,
        ez_c=zeros,
        hx_c=zeros,
        hy_c=zeros,
        hz_c=zeros,
        ex_f=zeros,
        ey_f=zeros,
        ez_f=zeros,
        hx_f=zeros,
        hy_f=zeros,
        hz_f=zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )
    sheet = _PrivateAnalyticSheetSourceSpec(
        name="synthetic_sheet",
        axis=2,
        index=3,
        component="ey",
        propagation_sign=1,
        amplitude=1.0,
        f0_hz=2.0e9,
        bandwidth=0.8,
        phase_rad=0.0,
        source_values=jnp.ones((4,), dtype=jnp.float64),
        lo1=1,
        hi1=4,
        lo2=2,
        hi2=5,
    )

    out = _inject_private_analytic_sheet_source(state, sheet, jnp.array(2.5))

    ey = np.asarray(out.ey_f)
    assert np.allclose(ey[1:4, 2:5, 3], 2.5)
    assert np.count_nonzero(ey) == 9
    assert np.allclose(np.asarray(out.ex_f), 0.0)


def test_private_tfsf_incident_ex_hy_signs_produce_positive_z_poynting():
    shape = (6, 6, 6)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    state = SubgridState3D(
        ex_c=zeros,
        ey_c=zeros,
        ez_c=zeros,
        hx_c=zeros,
        hy_c=zeros,
        hz_c=zeros,
        ex_f=zeros,
        ey_f=zeros,
        ez_f=zeros,
        hx_f=zeros,
        hy_f=zeros,
        hz_f=zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )
    incident = _PrivateTFSFIncidentSpec(
        name="synthetic_tfsf",
        axis=2,
        index=3,
        electric_component="ex",
        magnetic_component="hy",
        propagation_sign=1,
        electric_values=jnp.asarray([2.0], dtype=jnp.float64),
        magnetic_values=jnp.asarray([0.5], dtype=jnp.float64),
        lo1=1,
        hi1=5,
        lo2=1,
        hi2=5,
    )

    with_h = _apply_private_tfsf_incident_h(state, incident, jnp.array(0.5))
    out = _apply_private_tfsf_incident_e(with_h, incident, jnp.array(2.0))

    ex = np.asarray(out.ex_f)
    hy = np.asarray(out.hy_f)
    assert np.allclose(ex[1:5, 1:5, 3], 2.0)
    assert np.allclose(hy[1:5, 1:5, 2], 0.5)
    assert np.count_nonzero(ex) == 16
    assert np.count_nonzero(hy) == 16
    assert float(np.mean(ex[1:5, 1:5, 3] * hy[1:5, 1:5, 2])) > 0.0


def test_private_plane_rejects_local_normal_index_zero():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(
            _benchmark_plane("at_interface", coordinate=_FluxFixture.offsets[2])
        )


def test_private_plane_rejects_local_normal_index_n_minus_1():
    last_index_coordinate = (
        _FluxFixture.offsets[2] + (_FluxFixture.shape_f[2] - 1) * _FluxFixture.dx_f
    )

    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(
            _benchmark_plane("at_last_slice", coordinate=last_index_coordinate)
        )


def test_private_plane_rejects_plane_outside_fine_region():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(_benchmark_plane("outside", coordinate=0.010))


def test_private_plane_rejects_plane_not_fully_fine_owned():
    with pytest.raises(ValueError, match=_STRICT_PLACEMENT):
        _plane_specs(
            _benchmark_plane(
                "crosses_tangential_interface",
                coordinate=_FluxFixture.front_plane,
                size=(0.020, 0.010),
            )
        )


def _synthetic_subgrid_state(shape: tuple[int, int, int]) -> SubgridState3D:
    size = int(np.prod(shape))
    base = jnp.arange(size, dtype=jnp.float64).reshape(shape)

    def field(offset: float) -> jnp.ndarray:
        return (base + offset) / 100.0

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return SubgridState3D(
        ex_c=zeros,
        ey_c=zeros,
        ez_c=zeros,
        hx_c=zeros,
        hy_c=zeros,
        hz_c=zeros,
        ex_f=field(1.0),
        ey_f=field(2.0),
        ez_f=field(3.0),
        hx_f=field(4.0),
        hy_f=field(5.0),
        hz_f=field(6.0),
        step=jnp.array(3, dtype=jnp.int32),
    )


def _axis_plane_slices(axis: int, index: int, lo1: int, hi1: int, lo2: int, hi2: int):
    if axis == 0:
        return (
            (index, slice(lo1, hi1), slice(lo2, hi2)),
            (index - 1, slice(lo1, hi1), slice(lo2, hi2)),
        )
    if axis == 1:
        return (
            (slice(lo1, hi1), index, slice(lo2, hi2)),
            (slice(lo1, hi1), index - 1, slice(lo2, hi2)),
        )
    return (
        (slice(lo1, hi1), slice(lo2, hi2), index),
        (slice(lo1, hi1), slice(lo2, hi2), index - 1),
    )


def _reference_accumulate(
    acc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    state: SubgridState3D,
    plane: _BenchmarkFluxPlaneSpec,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = {
        0: ("ey_f", "ez_f", "hy_f", "hz_f"),
        1: ("ez_f", "ex_f", "hz_f", "hx_f"),
        2: ("ex_f", "ey_f", "hx_f", "hy_f"),
    }[plane.axis]
    idx, idx_m1 = _axis_plane_slices(
        plane.axis,
        plane.index,
        plane.lo1,
        plane.hi1,
        plane.lo2,
        plane.hi2,
    )
    e1 = np.asarray(getattr(state, names[0]))[idx]
    e2 = np.asarray(getattr(state, names[1]))[idx]
    h1 = 0.5 * (
        np.asarray(getattr(state, names[2]))[idx_m1]
        + np.asarray(getattr(state, names[2]))[idx]
    )
    h2 = 0.5 * (
        np.asarray(getattr(state, names[3]))[idx_m1]
        + np.asarray(getattr(state, names[3]))[idx]
    )
    step = np.asarray(state.step, dtype=np.float64)
    t = step * dt
    freqs = np.asarray(plane.freqs, dtype=np.float64)
    weight = float(
        np.asarray(
            dft_window_weight(
                state.step,
                plane.total_steps,
                plane.window,
                plane.window_alpha,
            )
        )
    )
    kernel_e = np.exp(-1j * 2.0 * np.pi * freqs * t)[:, None, None] * dt * weight
    kernel_h = (
        np.exp(-1j * 2.0 * np.pi * freqs * (t - dt * 0.5))[:, None, None] * dt * weight
    )
    return (
        acc[0] + e1[None, :, :] * kernel_e,
        acc[1] + e2[None, :, :] * kernel_e,
        acc[2] + h1[None, :, :] * kernel_h,
        acc[3] + h2[None, :, :] * kernel_h,
    )


def test_private_flux_accumulator_matches_uniform_scan_kernel_formula():
    shape = (6, 5, 4)
    dt = 1.25e-12
    freqs = jnp.asarray([1.5e9, 2.0e9], dtype=jnp.float64)
    plane = _BenchmarkFluxPlaneSpec(
        name="synthetic",
        axis=2,
        index=2,
        freqs=freqs,
        dx=1.0e-3,
        total_steps=8,
        lo1=1,
        hi1=5,
        lo2=1,
        hi2=4,
    )
    state = _synthetic_subgrid_state(shape)
    acc_shape = (len(freqs), plane.hi1 - plane.lo1, plane.hi2 - plane.lo2)
    acc0 = tuple(jnp.zeros(acc_shape, dtype=jnp.complex128) for _ in range(4))

    actual = _accumulate_benchmark_flux_plane(acc0, state, plane, dt)
    t = np.asarray(state.step, dtype=np.float64) * dt
    phase_e = np.exp(-1j * 2.0 * np.pi * np.asarray(freqs) * t)
    phase_h = np.exp(-1j * 2.0 * np.pi * np.asarray(freqs) * (t - dt * 0.5))
    kernel_e = phase_e[:, None, None] * dt
    kernel_h = phase_h[:, None, None] * dt

    plane_slice = (slice(plane.lo1, plane.hi1), slice(plane.lo2, plane.hi2))
    plane_idx = plane_slice + (plane.index,)
    plane_idx_m1 = plane_slice + (plane.index - 1,)
    ex = np.asarray(state.ex_f)[plane_idx]
    ey = np.asarray(state.ey_f)[plane_idx]
    hx = 0.5 * (
        np.asarray(state.hx_f)[plane_idx_m1] + np.asarray(state.hx_f)[plane_idx]
    )
    hy = 0.5 * (
        np.asarray(state.hy_f)[plane_idx_m1] + np.asarray(state.hy_f)[plane_idx]
    )
    expected = (
        ex[None, :, :] * kernel_e,
        ey[None, :, :] * kernel_e,
        hx[None, :, :] * kernel_h,
        hy[None, :, :] * kernel_h,
    )

    for actual_arr, expected_arr in zip(actual, expected):
        assert np.allclose(
            np.asarray(actual_arr),
            expected_arr,
            rtol=1e-6,
            atol=1e-9,
        )

    actual_plane = _BenchmarkFluxPlaneResult(
        name=plane.name,
        axis=plane.axis,
        index=plane.index,
        freqs=plane.freqs,
        dx=plane.dx,
        e1_dft=actual[0],
        e2_dft=actual[1],
        h1_dft=actual[2],
        h2_dft=actual[3],
        lo1=plane.lo1,
        hi1=plane.hi1,
        lo2=plane.lo2,
        hi2=plane.hi2,
    )
    expected_plane = actual_plane._replace(
        e1_dft=jnp.asarray(expected[0]),
        e2_dft=jnp.asarray(expected[1]),
        h1_dft=jnp.asarray(expected[2]),
        h2_dft=jnp.asarray(expected[3]),
    )
    assert np.allclose(
        np.asarray(_benchmark_flux_spectrum(actual_plane)),
        np.asarray(_benchmark_flux_spectrum(expected_plane)),
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("axis", "index", "lo1", "hi1", "lo2", "hi2"),
    [
        (0, 3, 1, 4, 1, 3),
        (1, 3, 1, 5, 1, 3),
        (2, 2, 1, 5, 1, 4),
    ],
)
def test_private_flux_accumulator_matches_multistep_all_axis_windowed_formula(
    axis: int,
    index: int,
    lo1: int,
    hi1: int,
    lo2: int,
    hi2: int,
):
    shape = (6, 5, 4)
    dt = 1.25e-12
    freqs = jnp.asarray([1.5e9, 2.0e9], dtype=jnp.float64)
    plane = _BenchmarkFluxPlaneSpec(
        name=f"synthetic_axis_{axis}",
        axis=axis,
        index=index,
        freqs=freqs,
        dx=1.0e-3,
        total_steps=9,
        window="hann",
        lo1=lo1,
        hi1=hi1,
        lo2=lo2,
        hi2=hi2,
    )
    acc_shape = (len(freqs), hi1 - lo1, hi2 - lo2)
    actual = tuple(jnp.zeros(acc_shape, dtype=jnp.complex128) for _ in range(4))
    expected = tuple(np.zeros(acc_shape, dtype=np.complex128) for _ in range(4))

    for step in (3, 4):
        state = _synthetic_subgrid_state(shape)._replace(
            step=jnp.array(step, dtype=jnp.int32)
        )
        actual = _accumulate_benchmark_flux_plane(actual, state, plane, dt)
        expected = _reference_accumulate(expected, state, plane, dt)

    for actual_arr, expected_arr in zip(actual, expected):
        assert np.allclose(
            np.asarray(actual_arr),
            expected_arr,
            rtol=1e-6,
            atol=1e-9,
        )


def _complex_flux(plane) -> np.ndarray:
    return np.asarray(
        jnp.sum(
            plane.e1_dft * jnp.conj(plane.h2_dft)
            - plane.e2_dft * jnp.conj(plane.h1_dft),
            axis=(-2, -1),
        )
        * (plane.dx * plane.dx)
    )


def _phase_error_deg(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs((np.angle(test / ref, deg=True) + 180.0) % 360.0 - 180.0)


def _relative_magnitude_error(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mag_test = np.abs(test)
    mag_ref = np.abs(ref)
    return np.where(
        mag_ref >= 1e-3,
        np.abs(mag_test - mag_ref) / np.maximum(mag_ref, _NORMALIZATION_FLOOR),
        np.abs(mag_test - mag_ref),
    )


class _FixtureRun(NamedTuple):
    dt: float
    complex_flux: tuple[np.ndarray, np.ndarray]
    signed_flux: tuple[np.ndarray, np.ndarray]
    planes: tuple[object, object]


def _fixture_run_from_private_benchmark(run) -> _FixtureRun:
    planes = run.benchmark_flux_planes
    signed_flux = tuple(np.asarray(_benchmark_flux_spectrum(p)) for p in planes)
    complex_flux = tuple(_complex_flux(p) for p in planes)
    return _FixtureRun(float(run.result.dt), complex_flux, signed_flux, planes)


def _finite_or_fail(label: str, values: np.ndarray) -> dict[str, object] | None:
    if not np.all(np.isfinite(values)):
        return {"classification": "fail", "reason": f"{label} contains NaN/Inf"}
    return None


def _plane_requests(
    shift_cells: int = 0,
    aperture_size: float | None = None,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> tuple[_BenchmarkFluxPlaneRequest, ...]:
    dz = shift_cells * fixture.dx_f
    size = fixture.aperture_size[0] if aperture_size is None else aperture_size
    aperture = (size, size)
    return (
        _benchmark_plane(
            "front",
            fixture=fixture,
            coordinate=fixture.front_plane + dz,
            size=aperture,
        ),
        _benchmark_plane(
            "back",
            fixture=fixture,
            coordinate=fixture.back_plane + dz,
            size=aperture,
        ),
    )


def _sheet_waveform(fixture: _FluxFixtureConfig = _FluxFixture) -> CustomWaveform:
    def waveform(t):
        tau = 1.0 / (jnp.pi * fixture.source_freq * fixture.source_bandwidth)
        t0 = 5.0 * tau
        envelope = jnp.exp(-(((t - t0) / tau) ** 2))
        carrier = jnp.sin(
            2.0 * jnp.pi * fixture.source_freq * t + fixture.sheet_phase_rad
        )
        return jnp.asarray(
            fixture.sheet_amplitude * carrier * envelope,
            dtype=jnp.float32,
        )

    return CustomWaveform(func=waveform)


def _sheet_axis_positions(span: tuple[float, float], dx: float) -> np.ndarray:
    n_cells = int(round((span[1] - span[0]) / dx))
    return np.asarray(span[0] + np.arange(n_cells, dtype=np.float64) * dx)


def _add_uniform_sheet_sources(
    sim: Simulation,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> None:
    waveform = _sheet_waveform(fixture)
    xs = _sheet_axis_positions(fixture.sheet_x_span, fixture.uniform_dx)
    ys = _sheet_axis_positions(fixture.sheet_y_span, fixture.uniform_dx)
    for x in xs:
        for y in ys:
            sim.add_source(
                position=(float(x), float(y), fixture.sheet_coordinate),
                component=fixture.sheet_component,
                waveform=waveform,
            )


@lru_cache(maxsize=None)
def _run_flux_fixture(
    *,
    subgrid: bool,
    slab: bool,
    fixture: _FluxFixtureConfig = _FluxFixture,
    plane_shift_cells: int = 0,
    aperture_size: float | None = None,
    source_kind: str = "analytic_sheet",
) -> _FixtureRun:
    if source_kind not in {"analytic_sheet", "private_tfsf"}:
        raise ValueError(f"unknown private benchmark source kind {source_kind!r}")

    dx = fixture.coarse_dx if subgrid else fixture.uniform_dx
    size = fixture.aperture_size[0] if aperture_size is None else aperture_size
    aperture = (size, size)
    sim = Simulation(
        freq_max=fixture.freq_max,
        domain=fixture.domain,
        boundary="cpml",
        cpml_layers=fixture.cpml_layers,
        dx=dx,
    )
    if slab:
        sim.add_material("rt_dielectric", eps_r=fixture.slab_eps_r)
        sim.add(Box(fixture.slab_lo, fixture.slab_hi), material="rt_dielectric")
    if subgrid:
        sim.add_refinement(**fixture.refinement)
    else:
        if source_kind == "analytic_sheet":
            _add_uniform_sheet_sources(sim, fixture)
    sim.add_probe(position=fixture.source, component=fixture.source_component)

    if subgrid:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No sources, ports, TFSF, or waveguide/Floquet ports configured.*",
                category=UserWarning,
            )
            run = run_subgridded_benchmark_flux(
                sim,
                n_steps=fixture.n_steps,
                planes=_plane_requests(plane_shift_cells, aperture_size, fixture),
                sheet_sources=(
                    (_benchmark_sheet_source(fixture=fixture),)
                    if source_kind == "analytic_sheet"
                    else ()
                ),
                private_tfsf_incidents=(
                    (_benchmark_tfsf_incident(fixture=fixture),)
                    if source_kind == "private_tfsf"
                    else ()
                ),
            )
        return _fixture_run_from_private_benchmark(run)

    if source_kind == "private_tfsf":
        run = run_private_tfsf_reference_flux(
            sim,
            n_steps=fixture.n_steps,
            planes=_plane_requests(plane_shift_cells, aperture_size, fixture),
            private_tfsf_incidents=(_benchmark_tfsf_incident(fixture=fixture),),
        )
        return _fixture_run_from_private_benchmark(run)

    dz = plane_shift_cells * fixture.dx_f
    sim.add_flux_monitor(
        axis="z",
        coordinate=fixture.front_plane + dz,
        freqs=fixture.scored_freqs,
        size=aperture,
        center=fixture.aperture_center,
        name="front",
    )
    sim.add_flux_monitor(
        axis="z",
        coordinate=fixture.back_plane + dz,
        freqs=fixture.scored_freqs,
        size=aperture,
        center=fixture.aperture_center,
        name="back",
    )
    result = sim.run(n_steps=fixture.n_steps)
    monitors = (result.flux_monitors["front"], result.flux_monitors["back"])
    signed_flux = tuple(np.asarray(flux_spectrum(m)) for m in monitors)
    complex_flux = tuple(_complex_flux(m) for m in monitors)
    return _FixtureRun(float(result.dt), complex_flux, signed_flux, monitors)


def _usable_passband(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    front_mag = np.abs(front)
    back_mag = np.abs(back)
    front_peak = float(np.max(front_mag))
    back_peak = float(np.max(back_mag))
    if front_peak <= 0.0 or back_peak <= 0.0:
        return np.zeros_like(front_mag, dtype=bool)
    return (front_mag >= 0.20 * front_peak) & (back_mag >= 0.20 * back_peak)


def _claims_bearing_passband(
    complex_flux: tuple[np.ndarray, np.ndarray],
    signed_flux: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mask = _usable_passband(complex_flux[0], complex_flux[1])
    nonfloor = _NONFLOOR_FACTOR * _NORMALIZATION_FLOOR
    for complex_values, signed_values in zip(complex_flux, signed_flux, strict=True):
        mask = (
            mask
            & (np.abs(complex_values) >= nonfloor)
            & (np.abs(signed_values) >= nonfloor)
        )
    return mask


def _sheet_component_dft(
    plane,
    fixture: _FluxFixtureConfig = _FluxFixture,
    component: str | None = None,
) -> np.ndarray:
    component = fixture.sheet_component if component is None else component
    if component == "ex":
        return np.asarray(plane.e1_dft)
    if component == "ey":
        return np.asarray(plane.e2_dft)
    raise ValueError(f"unsupported private benchmark field component {component!r}")


def _transverse_uniformity_metadata(
    planes: tuple[object, object],
    mask: np.ndarray,
    fixture: _FluxFixtureConfig = _FluxFixture,
    component: str | None = None,
) -> dict[str, object]:
    per_plane = []
    max_cv = 0.0
    max_phase_spread = 0.0
    if int(np.sum(mask)) == 0:
        return {
            "passed": False,
            "max_magnitude_cv": float("inf"),
            "max_phase_spread_deg": float("inf"),
            "per_plane": per_plane,
        }

    for plane_name, plane in zip(("front", "back"), planes, strict=True):
        field = _sheet_component_dft(plane, fixture, component)[mask]
        for freq_hz, values in zip(
            fixture.scored_freqs[mask],
            field,
            strict=True,
        ):
            mags = np.abs(values)
            mean_mag = float(np.mean(mags))
            cv = (
                float(np.std(mags) / mean_mag)
                if mean_mag > _NORMALIZATION_FLOOR
                else float("inf")
            )
            mean_complex = complex(np.mean(values))
            if abs(mean_complex) > _NORMALIZATION_FLOOR:
                phase_delta = np.angle(values * np.conj(mean_complex), deg=True)
                phase_spread = float(np.max(np.abs(phase_delta)))
            else:
                phase_spread = float("inf")
            max_cv = max(max_cv, cv)
            max_phase_spread = max(max_phase_spread, phase_spread)
            per_plane.append(
                {
                    "plane": plane_name,
                    "freq_hz": float(freq_hz),
                    "magnitude_cv": cv,
                    "phase_spread_deg": phase_spread,
                }
            )

    return {
        "passed": bool(
            max_cv <= _TRANSVERSE_MAGNITUDE_CV_MAX
            and max_phase_spread <= _TRANSVERSE_PHASE_SPREAD_DEG_MAX
        ),
        "max_magnitude_cv": max_cv,
        "max_phase_spread_deg": max_phase_spread,
        "per_plane": per_plane,
    }


def _vacuum_stability_metadata(
    uniform_flux: tuple[np.ndarray, np.ndarray],
    subgrid_flux: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> dict[str, object]:
    if int(np.sum(mask)) == 0:
        return {
            "passed": False,
            "max_magnitude_error": float("inf"),
            "max_phase_error_deg": float("inf"),
        }
    ref = np.concatenate([uniform_flux[0][mask], uniform_flux[1][mask]])
    sub = np.concatenate([subgrid_flux[0][mask], subgrid_flux[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    return {
        "passed": bool(
            float(np.max(mag_error)) <= _VACUUM_MAGNITUDE_ERROR_MAX
            and float(np.max(phase_error)) <= _VACUUM_PHASE_ERROR_DEG_MAX
        ),
        "max_magnitude_error": float(np.max(mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
    }


def _reference_quality_thresholds() -> dict[str, dict[str, float | int]]:
    return {
        "usable_passband": {
            "min_bins": _MIN_CLAIMS_BEARING_BINS,
            "front_back_peak_fraction": 0.20,
            "nonfloor_factor_times_normalization_floor": _NONFLOOR_FACTOR,
        },
        "transverse_uniformity": {
            "magnitude_cv_max": _TRANSVERSE_MAGNITUDE_CV_MAX,
            "phase_spread_deg_max": _TRANSVERSE_PHASE_SPREAD_DEG_MAX,
        },
        "vacuum_stability": {
            "relative_magnitude_error_max": _VACUUM_MAGNITUDE_ERROR_MAX,
            "phase_error_deg_max": _VACUUM_PHASE_ERROR_DEG_MAX,
        },
    }


def _reference_quality_thresholds_checksum(
    thresholds: dict[str, dict[str, float | int]] | None = None,
) -> str:
    payload = json.dumps(
        _reference_quality_thresholds() if thresholds is None else thresholds,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _causal_truth_table() -> dict[str, dict[str, object]]:
    base_positive = ("positive_evidence",)
    base_guard = ("guard_evidence", "baseline_metrics", "candidate_metrics")
    return {
        "measurement_or_reference_helper": {
            "positive_evidence": base_positive,
            "guard_evidence": ("public_outputs_absent",),
            "same_run_repair_allowed": True,
        },
        "source_waveform_or_stagger": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard + ("rung1_measurement_self_oracle_passed",),
            "same_run_repair_allowed": True,
        },
        "plane_placement_or_phase_center": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard
            + ("rung1_measurement_self_oracle_passed", "rung2_source_checked"),
            "same_run_repair_allowed": True,
        },
        "finite_aperture_source_edge": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard + ("full_aperture_metrics_visible",),
            "same_run_repair_allowed": True,
        },
        "aperture_edge_plus_interface_or_amplitude": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard + ("full_aperture_metrics_visible",),
            "same_run_repair_allowed": False,
        },
        "fixture_interface_geometry": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard + ("rungs_1_to_4_not_sufficient",),
            "same_run_repair_allowed": True,
        },
        "sbp_sat_interface_floor": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard
            + ("source_plane_aperture_controls_not_material",),
            "same_run_repair_allowed": False,
        },
        "private_hook_order_or_stagger": {
            "positive_evidence": base_positive,
            "guard_evidence": base_guard + ("hook_contingency_justification",),
            "same_run_repair_allowed": False,
        },
        "unresolved_after_ladder": {
            "positive_evidence": ("conflicting_or_insufficient_evidence",),
            "guard_evidence": ("all_entered_rungs_recorded",),
            "same_run_repair_allowed": False,
        },
    }


def _reference_quality_metrics(
    *,
    usable_bins: int,
    uniformity: dict[str, object],
    vacuum_stability: dict[str, object],
    source_eta0_consistency: dict[str, object] | None = None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "transverse_phase_spread_deg": float(uniformity["max_phase_spread_deg"]),
        "transverse_magnitude_cv": float(uniformity["max_magnitude_cv"]),
        "vacuum_relative_magnitude_error": float(
            vacuum_stability["max_magnitude_error"]
        ),
        "vacuum_phase_error_deg": float(vacuum_stability["max_phase_error_deg"]),
        "usable_bins": int(usable_bins),
    }
    if source_eta0_consistency is not None:
        metrics["source_eta0_relative_error"] = float(
            source_eta0_consistency["relative_error"]
        )
    return metrics


def _metric_threshold(metric: str) -> float:
    thresholds = {
        "transverse_phase_spread_deg": _TRANSVERSE_PHASE_SPREAD_DEG_MAX,
        "transverse_magnitude_cv": _TRANSVERSE_MAGNITUDE_CV_MAX,
        "vacuum_relative_magnitude_error": _VACUUM_MAGNITUDE_ERROR_MAX,
        "vacuum_phase_error_deg": _VACUUM_PHASE_ERROR_DEG_MAX,
        "source_eta0_relative_error": _VACUUM_MAGNITUDE_ERROR_MAX,
    }
    return float(thresholds[metric])


def _paired_metrics_for(dominant_metric: str) -> tuple[str, ...]:
    if dominant_metric == "transverse_phase_spread_deg":
        return ("transverse_magnitude_cv", "vacuum_phase_error_deg")
    if dominant_metric == "transverse_magnitude_cv":
        return ("transverse_phase_spread_deg", "vacuum_relative_magnitude_error")
    if dominant_metric == "vacuum_phase_error_deg":
        return ("vacuum_relative_magnitude_error", "transverse_phase_spread_deg")
    if dominant_metric == "vacuum_relative_magnitude_error":
        return ("vacuum_phase_error_deg", "transverse_magnitude_cv")
    if dominant_metric == "source_eta0_relative_error":
        return ("vacuum_relative_magnitude_error", "vacuum_phase_error_deg")
    return ()


def _relative_metric_improvement(before: float, after: float) -> float:
    if not np.isfinite(before) and np.isfinite(after):
        return 1.0
    if not np.isfinite(before):
        return 0.0
    if abs(before) <= _NORMALIZATION_FLOOR:
        return 0.0 if after >= before else 1.0
    return float((before - after) / abs(before))


def _material_improvement_decision(
    *,
    baseline_metrics: dict[str, float | int],
    candidate_metrics: dict[str, float | int],
    dominant_metric: str,
    thresholds_checksum: str | None = None,
    expected_thresholds_checksum: str | None = None,
) -> dict[str, object]:
    """Apply the causal-ladder 50%/25% anti-cherry-pick rule."""

    expected = _reference_quality_thresholds_checksum()
    checksum = expected if thresholds_checksum is None else thresholds_checksum
    expected_checksum = (
        expected
        if expected_thresholds_checksum is None
        else expected_thresholds_checksum
    )
    checksum_matches = checksum == expected_checksum

    dominant_before = float(baseline_metrics[dominant_metric])
    dominant_after = float(candidate_metrics[dominant_metric])
    dominant_improvement = _relative_metric_improvement(
        dominant_before,
        dominant_after,
    )
    dominant_threshold_crossed = dominant_after <= _metric_threshold(dominant_metric)
    dominant_ok = (
        dominant_improvement >= _DOMINANT_IMPROVEMENT_MIN or dominant_threshold_crossed
    )

    paired_results = {}
    paired_ok = False
    for paired_metric in _paired_metrics_for(dominant_metric):
        if (
            paired_metric not in baseline_metrics
            or paired_metric not in candidate_metrics
        ):
            continue
        before = float(baseline_metrics[paired_metric])
        after = float(candidate_metrics[paired_metric])
        improvement = _relative_metric_improvement(before, after)
        threshold_crossed = after <= _metric_threshold(paired_metric)
        ok = improvement >= _PAIRED_IMPROVEMENT_MIN or threshold_crossed
        paired_results[paired_metric] = {
            "baseline": before,
            "candidate": after,
            "relative_improvement": improvement,
            "threshold_crossed": bool(threshold_crossed),
            "passed": bool(ok),
        }
        paired_ok = paired_ok or ok

    usable_bins_ok = (
        int(candidate_metrics.get("usable_bins", 0)) >= _MIN_CLAIMS_BEARING_BINS
    )
    new_blocker_regressions = []
    for metric, before_value in baseline_metrics.items():
        if metric in {dominant_metric, "usable_bins"}:
            continue
        if metric not in candidate_metrics:
            continue
        before = float(before_value)
        after = float(candidate_metrics[metric])
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        if (
            before > _NORMALIZATION_FLOOR
            and after > _metric_threshold(metric)
            and (after - before) / before > _NEW_BLOCKER_REGRESSION_MAX
        ):
            new_blocker_regressions.append(
                {
                    "metric": metric,
                    "baseline": before,
                    "candidate": after,
                    "relative_regression": float((after - before) / before),
                }
            )

    passed = bool(
        checksum_matches
        and dominant_ok
        and paired_ok
        and usable_bins_ok
        and not new_blocker_regressions
    )
    if not checksum_matches:
        decision = "threshold_mismatch_inconclusive"
    elif new_blocker_regressions:
        decision = "tradeoff_inconclusive"
    elif passed:
        decision = "causal_candidate"
    else:
        decision = "inconclusive"
    return {
        "version": 1,
        "dominant_improvement_min": _DOMINANT_IMPROVEMENT_MIN,
        "paired_improvement_min": _PAIRED_IMPROVEMENT_MIN,
        "new_blocker_regression_max": _NEW_BLOCKER_REGRESSION_MAX,
        "dominant_metric": dominant_metric,
        "dominant": {
            "baseline": dominant_before,
            "candidate": dominant_after,
            "relative_improvement": dominant_improvement,
            "threshold_crossed": bool(dominant_threshold_crossed),
            "passed": bool(dominant_ok),
        },
        "paired": paired_results,
        "paired_passed": bool(paired_ok),
        "usable_bins_passed": bool(usable_bins_ok),
        "thresholds_checksum": checksum,
        "expected_thresholds_checksum": expected_checksum,
        "thresholds_checksum_matches": bool(checksum_matches),
        "new_blocker_regressions": new_blocker_regressions,
        "passed": passed,
        "classification_decision": decision,
    }


def _causal_ladder_candidate_record(
    *,
    candidate_id: str,
    rung: str,
    parameters: dict[str, object],
    cheap_rationale: str,
    baseline_packet_id: str,
    before_metrics: dict[str, float | int],
    after_metrics: dict[str, float | int],
    classification_decision: dict[str, object],
    predeclared: bool = True,
    command_requirement: dict[str, object] | None = None,
    full_aperture_metrics: dict[str, float | int] | None = None,
    candidate_class: str | None = None,
) -> dict[str, object]:
    command_requirement = (
        {"slow_command_required": False, "slow_command": None}
        if command_requirement is None
        else command_requirement
    )
    record: dict[str, object] = {
        "candidate_id": candidate_id,
        "rung": rung,
        "predeclared": bool(predeclared),
        "parameters": parameters,
        "cheap_rationale": cheap_rationale,
        "baseline_packet_id": baseline_packet_id,
        "thresholds_checksum": _reference_quality_thresholds_checksum(),
        "command_requirement": command_requirement,
        "slow_command_required": bool(command_requirement["slow_command_required"]),
        "slow_command": command_requirement.get("slow_command"),
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "classification_decision": classification_decision,
    }
    if candidate_class is not None:
        record["candidate_class"] = candidate_class
    if full_aperture_metrics is not None:
        record["full_aperture_metrics"] = full_aperture_metrics
    return record


def _causal_candidate_classification(
    *,
    candidate: dict[str, object],
    intended_class: str,
) -> str:
    if not candidate.get("predeclared", False):
        return "inconclusive"
    decision = candidate["classification_decision"]
    if not isinstance(decision, dict) or not decision.get("passed", False):
        return "inconclusive"
    return intended_class


def _source_eta0_consistency_metadata(
    electric_values: np.ndarray,
    magnetic_values: np.ndarray,
    *,
    eta0: float,
    tolerance: float = _VACUUM_MAGNITUDE_ERROR_MAX,
) -> dict[str, object]:
    electric = np.asarray(electric_values, dtype=np.float64)
    magnetic = np.asarray(magnetic_values, dtype=np.float64)
    expected = electric / float(eta0)
    mask = np.abs(expected) > max(
        float(np.max(np.abs(expected))) * 1.0e-12,
        _NORMALIZATION_FLOOR,
    )
    if int(np.sum(mask)) == 0:
        relative_error = float("inf")
    else:
        relative_error = float(
            np.max(
                np.abs(np.abs(magnetic[mask]) - np.abs(expected[mask]))
                / np.maximum(np.abs(expected[mask]), _NORMALIZATION_FLOOR)
            )
        )
    return {
        "metric": "source_eta0_relative_error",
        "relative_error": relative_error,
        "threshold": float(tolerance),
        "passed": bool(relative_error <= tolerance),
        "formula": "max(abs(abs(H)-abs(E/eta0))/max(abs(E/eta0), floor))",
    }


def _default_hook_contingency_justification(
    *,
    per_rung_status: dict[str, str] | None = None,
    direct_invariant_test: dict[str, object] | None = None,
    fixed_candidate: str = "baseline_full_aperture",
) -> dict[str, object]:
    per_rung_status = {} if per_rung_status is None else per_rung_status
    required = ("rung1", "rung2", "rung3", "rung4", "rung5")
    rung_negative = all(per_rung_status.get(rung) == "negative" for rung in required)
    direct_failed = bool(
        direct_invariant_test is not None
        and direct_invariant_test.get("passed") is False
    )
    eligible = bool(rung_negative or direct_failed)
    return {
        "eligible": eligible,
        "negative_evidence_required": list(required),
        "per_rung_status": per_rung_status,
        "direct_invariant_test": direct_invariant_test,
        "expected": None
        if direct_invariant_test is None
        else direct_invariant_test.get("expected"),
        "actual": None
        if direct_invariant_test is None
        else direct_invariant_test.get("actual"),
        "fixed_candidate": fixed_candidate,
        "hypothesis": (
            "specific hook order/sign/index hypothesis required" if eligible else None
        ),
        "why_hook_scope_is_allowed": (
            "rungs 1-5 are negative or a direct invariant failed" if eligible else None
        ),
        "rollback_policy": "remove diagnostic if not materially improving",
    }


def _same_run_repair_allowed(causal_class: str) -> bool:
    return causal_class in _ALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES


def _row2_no_claim_for_causal_class(causal_class: str) -> dict[str, object]:
    return {
        "causal_class": causal_class,
        "reference_quality_ready": False,
        "fixture_quality_ready": False,
        "slab_rt_scored": False,
        "public_claim_allowed": False,
        "follow_up_recommendation": "open a narrower follow-up plan before claims-bearing repair",
    }


def _reference_quality_blocker_ranking(
    *,
    usable_bins: int,
    nonfloor_flux: bool,
    uniformity: dict[str, object],
    vacuum_stability: dict[str, object],
) -> list[dict[str, object]]:
    """Rank row-2 blockers without relaxing or reinterpreting gate thresholds."""

    candidates = [
        {
            "name": "transverse_phase_spread_deg",
            "gate": "transverse_uniformity",
            "observed": float(uniformity["max_phase_spread_deg"]),
            "threshold": _TRANSVERSE_PHASE_SPREAD_DEG_MAX,
            "passed": float(uniformity["max_phase_spread_deg"])
            <= _TRANSVERSE_PHASE_SPREAD_DEG_MAX,
            "diagnostic_hint": "finite-aperture edge diffraction, phase-front tilt, or plane-placement mismatch",
        },
        {
            "name": "vacuum_relative_magnitude_error",
            "gate": "vacuum_stability",
            "observed": float(vacuum_stability["max_magnitude_error"]),
            "threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
            "passed": float(vacuum_stability["max_magnitude_error"])
            <= _VACUUM_MAGNITUDE_ERROR_MAX,
            "diagnostic_hint": "uniform/subgrid amplitude mismatch or SBP-SAT interface reflection floor",
        },
        {
            "name": "transverse_magnitude_cv",
            "gate": "transverse_uniformity",
            "observed": float(uniformity["max_magnitude_cv"]),
            "threshold": _TRANSVERSE_MAGNITUDE_CV_MAX,
            "passed": float(uniformity["max_magnitude_cv"])
            <= _TRANSVERSE_MAGNITUDE_CV_MAX,
            "diagnostic_hint": "finite-aperture edge diffraction or scored-aperture/source-span mismatch",
        },
        {
            "name": "vacuum_phase_error_deg",
            "gate": "vacuum_stability",
            "observed": float(vacuum_stability["max_phase_error_deg"]),
            "threshold": _VACUUM_PHASE_ERROR_DEG_MAX,
            "passed": float(vacuum_stability["max_phase_error_deg"])
            <= _VACUUM_PHASE_ERROR_DEG_MAX,
            "diagnostic_hint": "source timing, plane placement, or interface phase mismatch",
        },
        {
            "name": "usable_passband_bins",
            "gate": "usable_passband",
            "observed": int(usable_bins),
            "threshold": _MIN_CLAIMS_BEARING_BINS,
            "passed": int(usable_bins) >= _MIN_CLAIMS_BEARING_BINS,
            "diagnostic_hint": "insufficient non-floor frequency bins",
        },
        {
            "name": "nonfloor_front_back_flux",
            "gate": "analytic_incident_consistency",
            "observed": bool(nonfloor_flux),
            "threshold": True,
            "passed": bool(nonfloor_flux),
            "diagnostic_hint": "front/back signed flux fell to the normalization floor",
        },
    ]
    ranked = []
    for entry in candidates:
        observed = entry["observed"]
        threshold = entry["threshold"]
        if isinstance(observed, bool):
            severity = 0.0 if observed is threshold else float("inf")
        elif entry["name"] == "usable_passband_bins":
            severity = (
                0.0 if entry["passed"] else float(threshold) / max(float(observed), 1.0)
            )
        else:
            severity = float(observed) / max(float(threshold), _NORMALIZATION_FLOOR)
        ranked.append(entry | {"severity_vs_threshold": float(severity)})
    return sorted(
        ranked,
        key=lambda item: (bool(item["passed"]), -float(item["severity_vs_threshold"])),
    )


def _dominant_reference_quality_blocker(
    blockers: list[dict[str, object]],
) -> str:
    for blocker in blockers:
        if not blocker["passed"]:
            return str(blocker["name"])
    return "none"


def _private_tfsf_candidate_metrics_from_runs(
    *,
    ref_run: _FixtureRun,
    sub_run: _FixtureRun,
    fixture: _FluxFixtureConfig,
) -> dict[str, object]:
    mask = _claims_bearing_passband(sub_run.complex_flux, sub_run.signed_flux)
    uniformity = _transverse_uniformity_metadata(
        sub_run.planes,
        mask,
        fixture,
        component="ex",
    )
    vacuum_stability = _vacuum_stability_metadata(
        ref_run.complex_flux,
        sub_run.complex_flux,
        mask,
    )
    front_signed = np.asarray(sub_run.signed_flux[0])
    back_signed = np.asarray(sub_run.signed_flux[1])
    usable_passband = int(np.sum(mask)) >= _MIN_CLAIMS_BEARING_BINS
    nonfloor_flux = bool(
        usable_passband
        and np.all(np.abs(front_signed[mask]) >= _NORMALIZATION_FLOOR)
        and np.all(np.abs(back_signed[mask]) >= _NORMALIZATION_FLOOR)
    )
    blockers = _reference_quality_blocker_ranking(
        usable_bins=int(np.sum(mask)),
        nonfloor_flux=nonfloor_flux,
        uniformity=uniformity,
        vacuum_stability=vacuum_stability,
    )
    gates = {
        "usable_passband": bool(usable_passband),
        "transverse_uniformity": bool(uniformity["passed"]),
        "analytic_incident_consistency": bool(nonfloor_flux),
        "same_contract_reference": True,
        "vacuum_stability": bool(vacuum_stability["passed"]),
    }
    return {
        "fixture_quality_gates": gates,
        "reference_quality_ready": bool(all(gates.values())),
        "usable_bins": int(np.sum(mask)),
        "transverse_uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "reference_quality_blockers": blockers,
        "dominant_reference_quality_blocker": _dominant_reference_quality_blocker(
            blockers
        ),
        "metrics": {
            "usable_bins": int(np.sum(mask)),
            "transverse_magnitude_cv": float(uniformity["max_magnitude_cv"]),
            "transverse_phase_spread_deg": float(uniformity["max_phase_spread_deg"]),
            "vacuum_relative_magnitude_error": float(
                vacuum_stability["max_magnitude_error"]
            ),
            "vacuum_phase_error_deg": float(vacuum_stability["max_phase_error_deg"]),
        },
    }


def _private_tfsf_candidate_metrics(
    *,
    plane_shift_cells: int,
    aperture_size: float,
) -> dict[str, object]:
    ref_run = _run_flux_fixture(
        subgrid=False,
        slab=False,
        fixture=_BoundaryExpandedFluxFixture,
        plane_shift_cells=plane_shift_cells,
        aperture_size=aperture_size,
        source_kind="private_tfsf",
    )
    sub_run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=_BoundaryExpandedFluxFixture,
        plane_shift_cells=plane_shift_cells,
        aperture_size=aperture_size,
        source_kind="private_tfsf",
    )
    return _private_tfsf_candidate_metrics_from_runs(
        ref_run=ref_run,
        sub_run=sub_run,
        fixture=_BoundaryExpandedFluxFixture,
    )


def _private_time_centered_helper_c3_record() -> dict[str, object]:
    metrics = _PRIVATE_TIME_CENTERED_HELPER_C3_ROLLBACK_METRICS
    uniformity = {
        "passed": False,
        "max_magnitude_cv": metrics["transverse_magnitude_cv"],
        "max_phase_spread_deg": metrics["transverse_phase_spread_deg"],
    }
    vacuum_stability = {
        "passed": False,
        "max_magnitude_error": metrics["vacuum_relative_magnitude_error"],
        "max_phase_error_deg": metrics["vacuum_phase_error_deg"],
    }
    blockers = _reference_quality_blocker_ranking(
        usable_bins=int(metrics["usable_bins"]),
        nonfloor_flux=True,
        uniformity=uniformity,
        vacuum_stability=vacuum_stability,
    )
    return {
        **_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_LADDER[3],
        "status": "evaluated_and_rolled_back",
        "fixture_quality_gates": {
            "usable_passband": True,
            "transverse_uniformity": False,
            "analytic_incident_consistency": True,
            "same_contract_reference": True,
            "vacuum_stability": False,
        },
        "reference_quality_ready": False,
        "fixture_quality_ready": False,
        "metrics": metrics,
        "reference_quality_blockers": blockers,
        "dominant_reference_quality_blocker": _dominant_reference_quality_blocker(
            blockers
        ),
        "result_authority": (
            "solver-local C3 failed unchanged original-fixture thresholds and was "
            "rolled back to 0.02"
        ),
        "rollback_required": True,
        "rollback_verified": True,
        "retained_solver_relaxation": 0.02,
    }


def _private_time_centered_helper_fixture_quality_recovery_metadata(
    *,
    baseline_snapshot: dict[str, object],
) -> dict[str, object]:
    c0_metrics = _private_tfsf_candidate_metrics_from_runs(
        ref_run=baseline_snapshot["ref_run"],
        sub_run=baseline_snapshot["run"],
        fixture=_BoundaryExpandedFluxFixture,
    )
    c0 = {
        **_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_LADDER[0],
        "status": "scored",
        "fixture_quality_ready": bool(c0_metrics["reference_quality_ready"]),
        "result_authority": "current original fixture can claim recovery only if all gates pass",
        **c0_metrics,
    }
    c1_metrics = _private_tfsf_candidate_metrics(
        plane_shift_cells=0,
        aperture_size=0.010,
    )
    c1 = {
        **_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_LADDER[1],
        "status": "scored",
        "fixture_quality_ready": bool(c1_metrics["reference_quality_ready"]),
        "result_authority": (
            "measurement-control pass cannot claim original fixture recovery"
        ),
        **c1_metrics,
    }
    c2_metrics = _private_tfsf_candidate_metrics(
        plane_shift_cells=1,
        aperture_size=0.020,
    )
    c2 = {
        **_PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_LADDER[2],
        "status": "scored",
        "fixture_quality_ready": bool(c2_metrics["reference_quality_ready"]),
        "result_authority": (
            "measurement-control pass cannot claim original fixture recovery"
        ),
        **c2_metrics,
    }
    c3 = _private_time_centered_helper_c3_record()
    candidates = [c0, c1, c2, c3]
    original_recovered = any(
        bool(candidate["can_claim_original_fixture_recovery"])
        and bool(candidate["reference_quality_ready"])
        for candidate in candidates
    )
    terminal_outcome = (
        "private_time_centered_helper_fixture_quality_recovered"
        if original_recovered
        else _PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_STATUS
    )
    selected = next(
        (
            candidate
            for candidate in candidates
            if bool(candidate["can_claim_original_fixture_recovery"])
            and bool(candidate["reference_quality_ready"])
        ),
        c0,
    )
    return {
        "status": terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "candidate_ladder_declared_before_slow_scoring": True,
        "candidate_count": len(candidates),
        "candidate_policy": (
            "finite C0/C1/C2/C3 ladder; no adaptive sweeps, no threshold changes, "
            "C1/C2 cannot claim original fixture recovery, and C3 is the only "
            "solver-touching candidate"
        ),
        "thresholds_checksum": _reference_quality_thresholds_checksum(),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_parameters": selected["parameters"],
        "solver_hunk_touched": True,
        "solver_hunk_retained": False,
        "current_fixture_metrics_retained": True,
        "slab_rt_private_only": True,
        "fixture_quality_ready": bool(original_recovered),
        "reference_quality_ready": bool(original_recovered),
        "public_claim_allowed": False,
        "public_observable_promoted": False,
        "promotion_candidate_ready": False,
        "hook_experiment_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "slab_rt_public_claim_allowed": False,
        "candidates": candidates,
        "current_fixture_metrics": c0["metrics"],
        "dominant_reference_quality_blocker": c0["dominant_reference_quality_blocker"],
        "reference_quality_blockers": c0["reference_quality_blockers"],
        "next_prerequisite": (
            _PRIVATE_TIME_CENTERED_HELPER_FIXTURE_RECOVERY_NEXT_PREREQUISITE
        ),
        "reason": (
            "C0 original fixture, C1 center-core measurement control, C2 one-cell "
            "downstream plane control, and C3 temporary 0.05 helper relaxation all "
            "failed unchanged fixture/reference thresholds; C3 was rolled back and "
            "public promotion remains closed"
        ),
    }


def _serial_complex(values: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in np.ravel(values)]


def _flux_diagnostics(
    complex_flux: tuple[np.ndarray, np.ndarray],
    signed_flux: tuple[np.ndarray, np.ndarray],
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> dict[str, list[dict[str, object]]]:
    diagnostics: dict[str, list[dict[str, object]]] = {}
    for name, complex_values, signed_values in zip(
        ("front", "back"),
        complex_flux,
        signed_flux,
        strict=True,
    ):
        magnitude = np.abs(complex_values)
        diagnostics[name] = [
            {
                "freq_hz": float(freq),
                "complex": complex_pair,
                "magnitude": float(mag),
                "magnitude_to_floor": float(mag / _NORMALIZATION_FLOOR),
                "phase_deg": float(phase),
                "signed_real_flux": float(signed),
                "signed_real_flux_to_floor": float(abs(signed) / _NORMALIZATION_FLOOR),
            }
            for freq, complex_pair, mag, phase, signed in zip(
                fixture.scored_freqs,
                _serial_complex(complex_values),
                magnitude,
                np.angle(complex_values, deg=True),
                signed_values,
                strict=True,
            )
        ]
    return diagnostics


def _interface_distance_margins(
    fixture: _FluxFixtureConfig,
) -> dict[str, float]:
    """Record source/plane distances to the nearest SBP-SAT interface."""

    distances = {
        "sheet_to_lower_z_interface_m": fixture.sheet_coordinate
        - fixture.refinement_z_range[0],
        "sheet_to_upper_z_interface_m": fixture.refinement_z_range[1]
        - fixture.sheet_coordinate,
        "front_plane_to_lower_z_interface_m": fixture.front_plane
        - fixture.refinement_z_range[0],
        "front_plane_to_upper_z_interface_m": fixture.refinement_z_range[1]
        - fixture.front_plane,
        "back_plane_to_lower_z_interface_m": fixture.back_plane
        - fixture.refinement_z_range[0],
        "back_plane_to_upper_z_interface_m": fixture.refinement_z_range[1]
        - fixture.back_plane,
        "sheet_x_min_to_interface_m": fixture.sheet_x_span[0]
        - fixture.refinement_x_range[0],
        "sheet_x_max_to_interface_m": fixture.refinement_x_range[1]
        - fixture.sheet_x_span[1],
        "sheet_y_min_to_interface_m": fixture.sheet_y_span[0]
        - fixture.refinement_y_range[0],
        "sheet_y_max_to_interface_m": fixture.refinement_y_range[1]
        - fixture.sheet_y_span[1],
    }
    distances["min_source_or_plane_to_interface_m"] = float(
        min(
            distances["sheet_to_lower_z_interface_m"],
            distances["sheet_to_upper_z_interface_m"],
            distances["front_plane_to_lower_z_interface_m"],
            distances["front_plane_to_upper_z_interface_m"],
            distances["back_plane_to_lower_z_interface_m"],
            distances["back_plane_to_upper_z_interface_m"],
            distances["sheet_x_min_to_interface_m"],
            distances["sheet_x_max_to_interface_m"],
            distances["sheet_y_min_to_interface_m"],
            distances["sheet_y_max_to_interface_m"],
        )
    )
    return {key: float(value) for key, value in distances.items()}


def _front_back_signed_flux_ratio(
    signed_flux: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    front = np.asarray(signed_flux[0], dtype=np.float64)[mask]
    back = np.asarray(signed_flux[1], dtype=np.float64)[mask]
    return back / np.maximum(np.abs(front), _NORMALIZATION_FLOOR)


def _front_back_ratio_residual(
    *,
    uniform_signed_flux: tuple[np.ndarray, np.ndarray],
    subgrid_signed_flux: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
    fixture: _FluxFixtureConfig,
) -> dict[str, object]:
    """Private coarse/fine front-back residual used by interface diagnostics."""

    mask = np.asarray(mask, dtype=bool)
    if int(np.sum(mask)) == 0:
        return {
            "front_back_ratio_formula": _FRONT_BACK_RATIO_FORMULA,
            "ratio_error_formula": _FRONT_BACK_RATIO_ERROR_FORMULA,
            "threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
            "valid": False,
            "reason": "no scored passband bins",
            "usable_bins": 0,
            "scored_freqs_hz": [],
            "uniform_front_back_ratio": [],
            "subgrid_front_back_ratio": [],
            "ratio_error_by_freq": [],
            "max_ratio_error": float("inf"),
            "median_ratio_error": float("inf"),
            "front_back_ratio_sign_consistent": False,
            "uniform_reference_self_error": 0.0,
            "uniform_reference_below_threshold": True,
        }

    uniform_ratio = _front_back_signed_flux_ratio(uniform_signed_flux, mask)
    subgrid_ratio = _front_back_signed_flux_ratio(subgrid_signed_flux, mask)
    ratio_error = np.abs(subgrid_ratio - uniform_ratio) / np.maximum(
        np.abs(uniform_ratio),
        _NORMALIZATION_FLOOR,
    )
    sign_consistent = bool(
        np.all(np.signbit(uniform_ratio) == np.signbit(subgrid_ratio))
    )
    return {
        "front_back_ratio_formula": _FRONT_BACK_RATIO_FORMULA,
        "ratio_error_formula": _FRONT_BACK_RATIO_ERROR_FORMULA,
        "threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
        "valid": True,
        "usable_bins": int(np.sum(mask)),
        "scored_freqs_hz": fixture.scored_freqs[mask].tolist(),
        "uniform_front_back_ratio": [float(value) for value in uniform_ratio],
        "subgrid_front_back_ratio": [float(value) for value in subgrid_ratio],
        "ratio_error_by_freq": [float(value) for value in ratio_error],
        "max_ratio_error": float(np.max(ratio_error)),
        "median_ratio_error": float(np.median(ratio_error)),
        "front_back_ratio_sign_consistent": sign_consistent,
        "uniform_reference_self_error": 0.0,
        "uniform_reference_below_threshold": True,
    }


def _interface_energy_transfer_summary(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize whether a private front/back residual is stable across variants."""

    valid = [
        candidate
        for candidate in candidates
        if candidate.get("status") == "scored"
        and bool(candidate["energy_transfer"]["valid"])
    ]
    max_errors = [
        float(candidate["energy_transfer"]["max_ratio_error"]) for candidate in valid
    ]
    if len(valid) < 2:
        return {
            "front_back_ratio_formula": _FRONT_BACK_RATIO_FORMULA,
            "ratio_error_formula": _FRONT_BACK_RATIO_ERROR_FORMULA,
            "threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
            "candidate_count": len(valid),
            "interface_residual_stable": False,
            "stable_residual_evidence": "insufficient_candidates",
            "max_ratio_error": max(max_errors) if max_errors else float("inf"),
            "median_ratio_error": float(np.median(max_errors))
            if max_errors
            else float("inf"),
            "uniform_reference_below_threshold": True,
        }

    baseline_error = max_errors[0]
    no_material_reduction = all(
        _relative_metric_improvement(baseline_error, error) < _PAIRED_IMPROVEMENT_MIN
        for error in max_errors[1:]
    )
    all_above_threshold = all(
        error > _VACUUM_MAGNITUDE_ERROR_MAX for error in max_errors
    )
    sign_consistent = all(
        bool(candidate["energy_transfer"]["front_back_ratio_sign_consistent"])
        for candidate in valid
    )
    stable = bool(all_above_threshold and no_material_reduction and sign_consistent)
    return {
        "front_back_ratio_formula": _FRONT_BACK_RATIO_FORMULA,
        "ratio_error_formula": _FRONT_BACK_RATIO_ERROR_FORMULA,
        "threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
        "candidate_count": len(valid),
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in valid],
        "max_ratio_error": float(max(max_errors)),
        "median_ratio_error": float(np.median(max_errors)),
        "ratio_error_by_candidate": {
            str(candidate["candidate_id"]): float(
                candidate["energy_transfer"]["max_ratio_error"]
            )
            for candidate in valid
        },
        "all_candidates_above_threshold": bool(all_above_threshold),
        "no_material_reduction_vs_baseline": bool(no_material_reduction),
        "front_back_ratio_sign_order_consistent": bool(sign_consistent),
        "interface_residual_stable": stable,
        "stable_residual_evidence": "stable_across_predeclared_candidates"
        if stable
        else "not_stable",
        "uniform_reference_self_error": 0.0,
        "uniform_reference_below_threshold": True,
    }


def _fixture_with_tau(fixture: _FluxFixtureConfig, tau: float) -> _FluxFixtureConfig:
    tau = float(tau)
    if np.isclose(tau, fixture.tau):
        return fixture
    label = str(tau).replace(".", "p")
    return replace(
        fixture,
        name=f"{fixture.name}_tau_{label}",
        fixture_key=f"{fixture.fixture_key}_tau_{label}",
        tau=tau,
    )


def _private_tfsf_quality_snapshot_from_runs(
    *,
    fixture: _FluxFixtureConfig,
    ref_run: _FixtureRun,
    run: _FixtureRun,
    source_eta0_consistency: dict[str, object] | None = None,
) -> dict[str, object]:
    freq_mask = _claims_bearing_passband(run.complex_flux, run.signed_flux)
    uniformity = _transverse_uniformity_metadata(
        run.planes,
        freq_mask,
        fixture,
        component="ex",
    )
    vacuum_stability = _vacuum_stability_metadata(
        ref_run.complex_flux,
        run.complex_flux,
        freq_mask,
    )
    source_eta0_consistency = (
        _private_tfsf_source_eta0_metadata(fixture)
        if source_eta0_consistency is None
        else source_eta0_consistency
    )
    metrics = _reference_quality_metrics(
        usable_bins=int(np.sum(freq_mask)),
        uniformity=uniformity,
        vacuum_stability=vacuum_stability,
        source_eta0_consistency=source_eta0_consistency,
    )
    return {
        "ref_run": ref_run,
        "run": run,
        "freq_mask": freq_mask,
        "uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "source_eta0_consistency": source_eta0_consistency,
        "metrics": metrics,
        "usable_bins": int(np.sum(freq_mask)),
    }


def _private_tfsf_subgrid_quality_snapshot(
    *,
    fixture: _FluxFixtureConfig,
    ref_run: _FixtureRun,
    source_eta0_consistency: dict[str, object],
) -> dict[str, object]:
    run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=fixture,
        source_kind="private_tfsf",
    )
    return _private_tfsf_quality_snapshot_from_runs(
        fixture=fixture,
        ref_run=ref_run,
        run=run,
        source_eta0_consistency=source_eta0_consistency,
    )


def _paired_metric_regressions(
    *,
    baseline_metrics: dict[str, float | int],
    candidate_metrics: dict[str, float | int],
) -> list[dict[str, object]]:
    regressions: list[dict[str, object]] = []
    for metric, before_value in baseline_metrics.items():
        if metric == "usable_bins" or metric not in candidate_metrics:
            continue
        before = float(before_value)
        after = float(candidate_metrics[metric])
        if not (np.isfinite(before) and np.isfinite(after)):
            continue
        if before <= _NORMALIZATION_FLOOR:
            continue
        relative_regression = (after - before) / before
        if (
            relative_regression > _NEW_BLOCKER_REGRESSION_MAX
            and after > _metric_threshold(metric)
        ):
            regressions.append(
                {
                    "metric": metric,
                    "baseline": before,
                    "candidate": after,
                    "relative_regression": float(relative_regression),
                    "threshold": _metric_threshold(metric),
                }
            )
    if int(candidate_metrics.get("usable_bins", 0)) < int(
        baseline_metrics.get("usable_bins", 0)
    ):
        regressions.append(
            {
                "metric": "usable_bins",
                "baseline": int(baseline_metrics.get("usable_bins", 0)),
                "candidate": int(candidate_metrics.get("usable_bins", 0)),
                "relative_regression": 1.0,
                "threshold": _MIN_CLAIMS_BEARING_BINS,
            }
        )
    return regressions


def _private_repair_candidate_record(
    *,
    candidate_id: str,
    hypothesis: str,
    fixture: _FluxFixtureConfig,
    snapshot: dict[str, object],
    baseline_metrics: dict[str, float | int],
    baseline_residual: dict[str, object],
    command_requirement: dict[str, object],
) -> dict[str, object]:
    residual = _front_back_ratio_residual(
        uniform_signed_flux=snapshot["ref_run"].signed_flux,
        subgrid_signed_flux=snapshot["run"].signed_flux,
        mask=snapshot["freq_mask"],
        fixture=fixture,
    )
    baseline_error = float(baseline_residual["max_ratio_error"])
    candidate_error = float(residual["max_ratio_error"])
    relative_improvement = _relative_metric_improvement(
        baseline_error,
        candidate_error,
    )
    paired_regressions = _paired_metric_regressions(
        baseline_metrics=baseline_metrics,
        candidate_metrics=snapshot["metrics"],
    )
    material_passed = bool(
        residual["valid"]
        and relative_improvement >= _DOMINANT_IMPROVEMENT_MIN
        and not paired_regressions
    )
    strong_passed = bool(
        residual["valid"]
        and candidate_error <= _VACUUM_MAGNITUDE_ERROR_MAX
        and not paired_regressions
    )
    return {
        "candidate_id": candidate_id,
        "repair_hypothesis": hypothesis,
        "status": "scored",
        "predeclared": True,
        "fixture_name": fixture.name,
        "fixture": fixture.fixture_key,
        "fixture_parameters": fixture.to_metadata(),
        "candidate_tau": float(fixture.tau),
        "baseline_max_ratio_error": baseline_error,
        "candidate_max_ratio_error": candidate_error,
        "relative_improvement": float(relative_improvement),
        "paired_metric_regressions": paired_regressions,
        "material_improvement_passed": material_passed,
        "strong_private_repair_passed": strong_passed,
        "energy_transfer": residual,
        "metrics": snapshot["metrics"],
        "command_requirement": command_requirement,
    }


def _private_repair_outcome_from_candidates(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    scored = [candidate for candidate in candidates if candidate["status"] == "scored"]
    if not scored:
        return {
            "private_energy_transfer_repair_status": "no_material_repair",
            "selected_repair_candidate_id": None,
            "accepted_private_repair": False,
            "next_prerequisite": _PRIVATE_REPAIR_NO_MATERIAL_NEXT_PREREQUISITE,
            "outcome_reason": "no scored private repair candidates",
        }
    selected = min(
        scored,
        key=lambda candidate: float(candidate["candidate_max_ratio_error"]),
    )
    if bool(selected["strong_private_repair_passed"]):
        status = "repaired_private_floor"
        next_prerequisite = _PRIVATE_REPAIR_REPAIRED_NEXT_PREREQUISITE
        reason = "selected private candidate reduced residual below threshold"
    elif bool(selected["material_improvement_passed"]):
        status = "accepted_private_candidate"
        next_prerequisite = _PRIVATE_REPAIR_ACCEPTED_NEXT_PREREQUISITE
        reason = "selected private candidate passed material-improvement gate"
    else:
        status = "no_material_repair"
        next_prerequisite = _PRIVATE_REPAIR_NO_MATERIAL_NEXT_PREREQUISITE
        reason = "no predeclared private candidate passed material-improvement gate"
    return {
        "private_energy_transfer_repair_status": status,
        "selected_repair_candidate_id": selected["candidate_id"],
        "accepted_private_repair": status != "no_material_repair",
        "next_prerequisite": next_prerequisite,
        "outcome_reason": reason,
    }


def _floor_relative_error(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(np.abs(test) - np.abs(ref)) / np.maximum(
        np.abs(ref),
        _NORMALIZATION_FLOOR,
    )


def _energy_residual(
    signed_slab: tuple[np.ndarray, np.ndarray],
    signed_vacuum: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    signed_front = np.asarray(signed_slab[0])[mask]
    signed_back = np.asarray(signed_slab[1])[mask]
    signed_incident = np.asarray(signed_vacuum[0])[mask]
    return np.abs(
        (signed_front - signed_back)
        / np.maximum(np.abs(signed_incident), _NORMALIZATION_FLOOR)
    )


def _unfloored_energy_residual(
    signed_slab: tuple[np.ndarray, np.ndarray],
    signed_vacuum: tuple[np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    signed_front = np.asarray(signed_slab[0])[mask]
    signed_back = np.asarray(signed_slab[1])[mask]
    signed_incident = np.asarray(signed_vacuum[0])[mask]
    denom = np.where(np.abs(signed_incident) > 0.0, np.abs(signed_incident), np.nan)
    return np.abs((signed_front - signed_back) / denom)


@lru_cache(maxsize=None)
def _homogeneous_parity_for_aperture(
    aperture_size: float,
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> dict[str, object]:
    aperture_arg = (
        None if np.isclose(aperture_size, fixture.aperture_size[0]) else aperture_size
    )
    ref_run = _run_flux_fixture(
        subgrid=False,
        slab=False,
        fixture=fixture,
        aperture_size=aperture_arg,
    )
    sub_run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=fixture,
        aperture_size=aperture_arg,
    )
    dt_ref, flux_ref = (
        ref_run.dt,
        ref_run.complex_flux,
    )
    dt_sub, flux_sub, signed_sub = (
        sub_run.dt,
        sub_run.complex_flux,
        sub_run.signed_flux,
    )
    metadata: dict[str, object] = {
        "aperture_size_m": float(aperture_size),
        "dt_match": bool(np.allclose(dt_ref, dt_sub)),
        "uniform_front_peak_magnitude": float(np.max(np.abs(flux_ref[0]))),
        "uniform_back_peak_magnitude": float(np.max(np.abs(flux_ref[1]))),
        "subgrid_front_peak_magnitude": float(np.max(np.abs(flux_sub[0]))),
        "subgrid_back_peak_magnitude": float(np.max(np.abs(flux_sub[1]))),
    }
    metadata["fixture_name"] = fixture.name
    metadata["fixture"] = fixture.fixture_key
    metadata["uniform_front_peak_to_floor"] = float(
        metadata["uniform_front_peak_magnitude"] / _NORMALIZATION_FLOOR
    )
    metadata["uniform_back_peak_to_floor"] = float(
        metadata["uniform_back_peak_magnitude"] / _NORMALIZATION_FLOOR
    )

    mask = _claims_bearing_passband(flux_sub, signed_sub)
    metadata["usable_bins"] = int(np.sum(mask))
    metadata["scored_freqs_hz"] = fixture.scored_freqs[mask].tolist()
    if int(np.sum(mask)) < _MIN_CLAIMS_BEARING_BINS:
        metadata["classification"] = "inconclusive"
        metadata["reason"] = "homogeneous runtime passband is too weak to score"
        metadata["source_contract"] = "private_analytic_sheet_source"
        metadata["normalization"] = "vacuum_device_two_run_incident_normalized"
        return metadata

    ref = np.concatenate([flux_ref[0][mask], flux_ref[1][mask]])
    sub = np.concatenate([flux_sub[0][mask], flux_sub[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    metadata.update(
        {
            "classification": "pass"
            if float(np.max(mag_error)) <= 0.02 and float(np.max(phase_error)) <= 2.0
            else "inconclusive",
            "max_floor_relative_magnitude_error": float(np.max(mag_error)),
            "max_complex_phase_error_deg": float(np.max(phase_error)),
        }
    )
    return metadata


@lru_cache(maxsize=None)
def _homogeneous_parity_metadata(
    fixture: _FluxFixtureConfig = _FluxFixture,
) -> dict[str, object]:
    ref_run = _run_flux_fixture(subgrid=False, slab=False, fixture=fixture)
    sub_run = _run_flux_fixture(subgrid=True, slab=False, fixture=fixture)
    dt_ref, flux_ref, signed_ref = (
        ref_run.dt,
        ref_run.complex_flux,
        ref_run.signed_flux,
    )
    dt_sub, flux_sub, signed_sub = (
        sub_run.dt,
        sub_run.complex_flux,
        sub_run.signed_flux,
    )
    if not np.allclose(dt_ref, dt_sub):
        return {"classification": "fail", "reason": "uniform/subgrid dt mismatch"}
    for label, arrays in {"uniform": flux_ref, "subgrid": flux_sub}.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail

    mask = _claims_bearing_passband(flux_sub, signed_sub)
    if int(np.sum(mask)) < _MIN_CLAIMS_BEARING_BINS:
        return {
            "classification": "inconclusive",
            "reason": "homogeneous runtime passband is too weak to score",
            "fixture": f"{fixture.fixture_key}_homogeneous",
            "fixture_name": fixture.name,
            "fixture_parameters": fixture.to_metadata(),
            "source_contract": "private_analytic_sheet_source",
            "normalization": "vacuum_device_two_run_incident_normalized",
            "front_abs": np.abs(flux_ref[0]).tolist(),
            "back_abs": np.abs(flux_ref[1]).tolist(),
            "usable_bins": int(np.sum(mask)),
            "no_go_reason": _NO_GO_REASON,
            "blocking_diagnostic": (
                "the private analytic sheet did not produce at least two "
                "non-floor homogeneous passband bins for uniform/subgrid "
                "vacuum parity"
            ),
            "next_prerequisite": _NEXT_PREREQUISITE,
        }

    ref = np.concatenate([flux_ref[0][mask], flux_ref[1][mask]])
    sub = np.concatenate([flux_sub[0][mask], flux_sub[1][mask]])
    mag_error = _floor_relative_error(sub, ref)
    phase_error = _phase_error_deg(sub, ref)
    gates = {
        "magnitude": float(np.max(mag_error)) <= 0.02,
        "phase": float(np.max(phase_error)) <= 2.0,
    }
    return {
        "classification": "pass" if all(gates.values()) else "inconclusive",
        "fixture": f"{fixture.fixture_key}_homogeneous",
        "fixture_name": fixture.name,
        "fixture_parameters": fixture.to_metadata(),
        "source_contract": "private_analytic_sheet_source",
        "normalization": "vacuum_device_two_run_incident_normalized",
        "gates": gates,
        "scored_freqs_hz": fixture.scored_freqs[mask].tolist(),
        "max_magnitude_error": float(np.max(mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "normalization_floor": _NORMALIZATION_FLOOR,
        "public_claim_allowed": False,
        "uniform_flux_diagnostics": _flux_diagnostics(flux_ref, signed_ref, fixture),
        "subgrid_flux_diagnostics": _flux_diagnostics(flux_sub, signed_sub, fixture),
        "aperture_sweep": [
            _homogeneous_parity_for_aperture(fixture.aperture_size[0], fixture),
            _homogeneous_parity_for_aperture(
                max(fixture.aperture_size[0] - 0.004, fixture.uniform_dx * 4),
                fixture,
            ),
        ],
        "no_go_reason": _NO_GO_REASON,
        "blocking_diagnostic": (
            "homogeneous vacuum stability remains below pass threshold for "
            "the private analytic-sheet incident field"
        ),
        "next_prerequisite": _NEXT_PREREQUISITE,
        "diagnostic_basis": (
            "Synthetic multi-step/all-axis accumulator parity passes, and "
            "runtime scoring now uses the private analytic sheet plus "
            "non-floor incident bins before any public claim can be considered."
        ),
    }


def _quality_error_score(metadata: dict[str, object]) -> float:
    uniformity = metadata["transverse_uniformity"]
    vacuum = metadata["vacuum_stability"]
    if metadata["usable_bins"] < _MIN_CLAIMS_BEARING_BINS:
        return float("inf")
    ratios = [
        float(uniformity["max_magnitude_cv"]) / 0.01,
        float(uniformity["max_phase_spread_deg"]) / 1.0,
        float(vacuum["max_magnitude_error"]) / 0.02,
        float(vacuum["max_phase_error_deg"]) / 2.0,
    ]
    return float(max(ratios))


@lru_cache(maxsize=None)
def _fixture_quality_metadata(
    fixture: _FluxFixtureConfig,
) -> dict[str, object]:
    ref_run = _run_flux_fixture(subgrid=False, slab=False, fixture=fixture)
    sub_run = _run_flux_fixture(subgrid=True, slab=False, fixture=fixture)
    if not np.allclose(ref_run.dt, sub_run.dt):
        return {
            "classification": "fail",
            "reason": "uniform/subgrid dt mismatch",
            "fixture_name": fixture.name,
            "fixture": fixture.fixture_key,
        }
    for label, arrays in {
        "uniform": ref_run.complex_flux,
        "subgrid": sub_run.complex_flux,
    }.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail | {
                    "fixture_name": fixture.name,
                    "fixture": fixture.fixture_key,
                }

    mask = _claims_bearing_passband(sub_run.complex_flux, sub_run.signed_flux)
    uniformity = _transverse_uniformity_metadata(sub_run.planes, mask, fixture)
    vacuum_stability = _vacuum_stability_metadata(
        ref_run.complex_flux,
        sub_run.complex_flux,
        mask,
    )
    gates = {
        "usable_passband": int(np.sum(mask)) >= _MIN_CLAIMS_BEARING_BINS,
        "transverse_uniformity": bool(uniformity["passed"]),
        "vacuum_stability": bool(vacuum_stability["passed"]),
    }
    metadata: dict[str, object] = {
        "classification": "pass" if all(gates.values()) else "inconclusive",
        "fixture_name": fixture.name,
        "fixture": fixture.fixture_key,
        "fixture_parameters": fixture.to_metadata(),
        "fixture_quality_gates": gates,
        "usable_bins": int(np.sum(mask)),
        "scored_freqs_hz": fixture.scored_freqs[mask].tolist(),
        "transverse_uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "uniform_flux_diagnostics": _flux_diagnostics(
            ref_run.complex_flux,
            ref_run.signed_flux,
            fixture,
        ),
        "subgrid_flux_diagnostics": _flux_diagnostics(
            sub_run.complex_flux,
            sub_run.signed_flux,
            fixture,
        ),
    }
    metadata["quality_error_score"] = _quality_error_score(metadata)
    return metadata


@lru_cache(maxsize=None)
def _boundary_expansion_sweep_metadata() -> dict[str, object]:
    candidates = [
        _fixture_quality_metadata(fixture) for fixture in _RECOVERY_SWEEP_FIXTURES
    ]
    baseline = candidates[0]
    best = min(
        candidates,
        key=lambda item: float(item.get("quality_error_score", float("inf"))),
    )
    baseline_score = float(baseline.get("quality_error_score", float("inf")))
    best_score = float(best.get("quality_error_score", float("inf")))
    materially_improved = (
        best["fixture_name"] != baseline["fixture_name"]
        and np.isfinite(best_score)
        and (not np.isfinite(baseline_score) or best_score <= 0.90 * baseline_score)
    )
    return {
        "status": "pass" if best["classification"] == "pass" else "inconclusive",
        "candidate_count": len(candidates),
        "baseline_fixture": baseline["fixture_name"],
        "selected_fixture": best["fixture_name"],
        "selected_fixture_key": best["fixture"],
        "materially_improved_vs_baseline": bool(materially_improved),
        "baseline_quality_error_score": baseline_score,
        "selected_quality_error_score": best_score,
        "candidates": candidates,
    }


def _private_tfsf_source_eta0_metadata(
    fixture: _FluxFixtureConfig,
) -> dict[str, object]:
    from rfx.core.yee import EPS_0, MU_0

    (incident,) = _tfsf_specs(
        _benchmark_tfsf_incident(fixture=fixture), fixture=fixture
    )
    return _source_eta0_consistency_metadata(
        np.asarray(incident.electric_values),
        np.asarray(incident.magnetic_values),
        eta0=float(np.sqrt(MU_0 / EPS_0)),
    )


def _private_tfsf_reference_quality_snapshot(
    *,
    fixture: _FluxFixtureConfig,
    plane_shift_cells: int = 0,
    aperture_size: float | None = None,
    source_eta0_consistency: dict[str, object] | None = None,
) -> dict[str, object]:
    ref_run = _run_flux_fixture(
        subgrid=False,
        slab=False,
        fixture=fixture,
        plane_shift_cells=plane_shift_cells,
        aperture_size=aperture_size,
        source_kind="private_tfsf",
    )
    run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=fixture,
        plane_shift_cells=plane_shift_cells,
        aperture_size=aperture_size,
        source_kind="private_tfsf",
    )
    return _private_tfsf_quality_snapshot_from_runs(
        fixture=fixture,
        ref_run=ref_run,
        run=run,
        source_eta0_consistency=source_eta0_consistency,
    )


def _private_tfsf_causal_ladder_metadata(
    *,
    fixture: _FluxFixtureConfig,
    baseline_metrics: dict[str, float | int],
    dominant_reference_quality_blocker: str,
    source_eta0_consistency: dict[str, object],
) -> dict[str, object]:
    thresholds_checksum = _reference_quality_thresholds_checksum()
    baseline_packet_id = (
        f"{fixture.name}:private_tfsf:full_aperture:{thresholds_checksum[:12]}"
    )
    slow_command = _PRIVATE_TRUE_RT_SLOW_COMMAND
    material_rule = {
        "version": 1,
        "dominant_improvement_min": _DOMINANT_IMPROVEMENT_MIN,
        "paired_improvement_min": _PAIRED_IMPROVEMENT_MIN,
        "new_blocker_regression_max": _NEW_BLOCKER_REGRESSION_MAX,
        "source_eta0_relative_error_threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
        "thresholds_checksum": thresholds_checksum,
    }
    baseline_decision = _material_improvement_decision(
        baseline_metrics=baseline_metrics,
        candidate_metrics=baseline_metrics,
        dominant_metric=dominant_reference_quality_blocker,
    )
    candidate_records = [
        _causal_ladder_candidate_record(
            candidate_id="rung0_baseline_full_aperture",
            rung="rung0_baseline_freeze",
            parameters={
                "fixture_name": fixture.name,
                "plane_shift_cells": 0,
                "aperture_size_m": fixture.aperture_size[0],
                "source_kind": "private_tfsf",
            },
            cheap_rationale=(
                "Freeze the current full-aperture row-2 packet before any "
                "candidate comparison."
            ),
            baseline_packet_id=baseline_packet_id,
            before_metrics=baseline_metrics,
            after_metrics=baseline_metrics,
            classification_decision=baseline_decision
            | {"classification_decision": "baseline_no_causal_claim"},
            command_requirement={
                "slow_command_required": False,
                "slow_command": None,
            },
            full_aperture_metrics=baseline_metrics,
            candidate_class="baseline",
        )
    ]

    plane_shift_snapshot = _private_tfsf_reference_quality_snapshot(
        fixture=fixture,
        plane_shift_cells=2,
        source_eta0_consistency=source_eta0_consistency,
    )
    plane_shift_decision = _material_improvement_decision(
        baseline_metrics=baseline_metrics,
        candidate_metrics=plane_shift_snapshot["metrics"],
        dominant_metric=dominant_reference_quality_blocker,
    )
    plane_shift_record = _causal_ladder_candidate_record(
        candidate_id="rung3_plane_shift_plus_2_cells",
        rung="rung3_plane_placement_phase_center",
        parameters={
            "plane_shift_cells": 2,
            "front_plane_m": fixture.front_plane + 2 * fixture.dx_f,
            "back_plane_m": fixture.back_plane + 2 * fixture.dx_f,
            "strict_interior_required": True,
        },
        cheap_rationale=(
            "A bounded strict-interior plane shift checks whether the phase "
            "spread is caused by near-field or phase-center placement."
        ),
        baseline_packet_id=baseline_packet_id,
        before_metrics=baseline_metrics,
        after_metrics=plane_shift_snapshot["metrics"],
        classification_decision=plane_shift_decision,
        command_requirement={
            "slow_command_required": True,
            "slow_command": slow_command,
        },
        full_aperture_metrics=baseline_metrics,
        candidate_class="plane_placement_or_phase_center",
    )
    candidate_records.append(plane_shift_record)

    core_aperture = max(fixture.aperture_size[0] - 0.004, fixture.uniform_dx * 4)
    aperture_snapshot = _private_tfsf_reference_quality_snapshot(
        fixture=fixture,
        aperture_size=core_aperture,
        source_eta0_consistency=source_eta0_consistency,
    )
    aperture_decision = _material_improvement_decision(
        baseline_metrics=baseline_metrics,
        candidate_metrics=aperture_snapshot["metrics"],
        dominant_metric=dominant_reference_quality_blocker,
    )
    aperture_class = (
        "finite_aperture_source_edge"
        if bool(aperture_snapshot["vacuum_stability"]["passed"])
        else "aperture_edge_plus_interface_or_amplitude"
    )
    aperture_record = _causal_ladder_candidate_record(
        candidate_id="rung4_central_core_aperture",
        rung="rung4_aperture_source_edge",
        parameters={
            "aperture_size_m": core_aperture,
            "full_aperture_size_m": fixture.aperture_size[0],
            "source_span_x": list(fixture.sheet_x_span),
            "source_span_y": list(fixture.sheet_y_span),
            "full_aperture_metrics_visible": True,
        },
        cheap_rationale=(
            "A central-core aperture checks whether full-aperture edge "
            "diffraction dominates the transverse phase spread."
        ),
        baseline_packet_id=baseline_packet_id,
        before_metrics=baseline_metrics,
        after_metrics=aperture_snapshot["metrics"],
        classification_decision=aperture_decision,
        command_requirement={
            "slow_command_required": True,
            "slow_command": slow_command,
        },
        full_aperture_metrics=baseline_metrics,
        candidate_class=aperture_class,
    )
    candidate_records.append(aperture_record)

    source_passed = bool(source_eta0_consistency["passed"])
    plane_class = _causal_candidate_classification(
        candidate=plane_shift_record,
        intended_class="plane_placement_or_phase_center",
    )
    aperture_candidate_class = _causal_candidate_classification(
        candidate=aperture_record,
        intended_class=aperture_class,
    )
    if plane_class != "inconclusive":
        causal_class = plane_class
    elif aperture_candidate_class != "inconclusive":
        causal_class = aperture_candidate_class
    elif source_passed:
        causal_class = "sbp_sat_interface_floor"
    else:
        causal_class = "unresolved_after_ladder"

    rungs = {
        "rung0_baseline_freeze": {
            "status": "complete",
            "candidate_id": "rung0_baseline_full_aperture",
            "baseline_packet_id": baseline_packet_id,
        },
        "rung1_measurement_dft_scoring_self_oracle": {
            "status": "guarded_by_existing_private_replay_and_no_public_leak_tests",
            "classification_decision": "not_measurement_or_reference_helper",
            "evidence": [
                "same-contract private reference helper is present",
                "private Result does not expose public DFT/flux observables",
                "private H/E slots are locked before CPML",
            ],
        },
        "rung2_source_waveform_stagger": {
            "status": "cheap_eta0_consistency_passed"
            if source_passed
            else "source_eta0_consistency_failed",
            "source_eta0_consistency": source_eta0_consistency,
        },
        "rung3_plane_placement_phase_center": {
            "status": "material_candidate"
            if plane_class != "inconclusive"
            else "no_material_improvement",
            "candidate_id": plane_shift_record["candidate_id"],
        },
        "rung4_aperture_source_edge": {
            "status": "material_candidate"
            if aperture_candidate_class != "inconclusive"
            else "no_material_improvement",
            "candidate_id": aperture_record["candidate_id"],
        },
        "rung5_interface_floor": {
            "status": "implicated"
            if causal_class == "sbp_sat_interface_floor"
            else "not_entered_or_not_implicated",
            "classification_decision": (
                "persistent row-2 error after source, plane, and aperture "
                "controls did not satisfy material-improvement gates"
                if causal_class == "sbp_sat_interface_floor"
                else "defer"
            ),
        },
        "rung6_hook_contingency": {
            "status": "closed_by_default",
        },
    }
    hook_per_rung_status = {
        "rung1": "negative",
        "rung2": "negative" if source_passed else "positive",
        "rung3": "negative" if plane_class == "inconclusive" else "positive",
        "rung4": "negative"
        if aperture_candidate_class == "inconclusive"
        else "positive",
        "rung5": "positive" if causal_class == "sbp_sat_interface_floor" else "pending",
    }
    return {
        "causal_ladder_status": (
            "row2_causal_classified"
            if causal_class != "unresolved_after_ladder"
            else "row2_unresolved_after_ladder"
        ),
        "causal_class": causal_class,
        "causal_ladder_rungs": rungs,
        "causal_ladder_candidates": candidate_records,
        "material_improvement_rule": material_rule,
        "hook_contingency_justification": _default_hook_contingency_justification(
            per_rung_status=hook_per_rung_status,
            fixed_candidate=baseline_packet_id,
        ),
        "same_run_repair_allowed": _same_run_repair_allowed(causal_class),
        "follow_up_recommendation": (
            "open a solver/interface-floor investigation before low-level hook "
            "experiments"
            if causal_class == "sbp_sat_interface_floor"
            else "use the material candidate class for the next narrow repair plan"
        ),
    }


def _interface_distance_candidate_record(
    *,
    candidate_id: str,
    role: str,
    fixture: _FluxFixtureConfig,
    snapshot: dict[str, object],
    baseline_metrics: dict[str, float | int],
    dominant_metric: str,
    baseline_packet_id: str,
    command_requirement: dict[str, object],
) -> dict[str, object]:
    decision = _material_improvement_decision(
        baseline_metrics=baseline_metrics,
        candidate_metrics=snapshot["metrics"],
        dominant_metric=dominant_metric,
    )
    energy_transfer = _front_back_ratio_residual(
        uniform_signed_flux=snapshot["ref_run"].signed_flux,
        subgrid_signed_flux=snapshot["run"].signed_flux,
        mask=snapshot["freq_mask"],
        fixture=fixture,
    )
    return {
        "candidate_id": candidate_id,
        "role": role,
        "status": "scored",
        "predeclared": True,
        "fixture_name": fixture.name,
        "fixture": fixture.fixture_key,
        "fixture_parameters": fixture.to_metadata(),
        "interface_distances_m": _interface_distance_margins(fixture),
        "usable_bins": int(snapshot["usable_bins"]),
        "metrics": snapshot["metrics"],
        "classification_decision": decision["classification_decision"],
        "material_improvement_decision": decision,
        "baseline_packet_id": baseline_packet_id,
        "thresholds_checksum": _reference_quality_thresholds_checksum(),
        "command_requirement": command_requirement,
        "energy_transfer": energy_transfer,
    }


def _private_direct_invariant_tests() -> list[dict[str, object]]:
    reference_source = inspect.getsource(run_private_tfsf_reference_flux)
    h_update = reference_source.index("state = update_h(")
    h_private = reference_source.index("state = _apply_private_h_all(state, step_idx)")
    h_cpml = reference_source.index("apply_cpml_h(")
    e_update = reference_source.index("state = update_e(")
    e_private = reference_source.index("state = _apply_private_e_all(state, step_idx)")
    e_cpml = reference_source.index("apply_cpml_e(")
    slot_order_passed = h_update < h_private < h_cpml < e_update < e_private < e_cpml

    planes = _plane_specs(
        *_plane_requests(fixture=_BoundaryExpandedFluxFixture),
        fixture=_BoundaryExpandedFluxFixture,
    )
    strict_indices_passed = all(
        1 <= plane.index <= _BoundaryExpandedFluxFixture.shape_f[2] - 2
        for plane in planes
    )

    return [
        {
            "name": "private_tfsf_positive_z_poynting",
            "passed": True,
            "expected": "+z ex/hy incident fields produce positive Poynting",
            "actual": "covered by test_private_tfsf_incident_ex_hy_signs_produce_positive_z_poynting",
            "hypothesis_if_failed": "private_tfsf_sign_or_component_mismatch",
        },
        {
            "name": "private_reference_pre_cpml_slot_order",
            "passed": bool(slot_order_passed),
            "expected": "update_h < private_h < cpml_h < update_e < private_e < cpml_e",
            "actual": "order matched" if slot_order_passed else "order mismatch",
            "hypothesis_if_failed": "private_reference_hook_order_mismatch",
        },
        {
            "name": "private_plane_normal_indices_strict_interior",
            "passed": bool(strict_indices_passed),
            "expected": "all private planes satisfy 1 <= normal index <= n-2",
            "actual": [int(plane.index) for plane in planes],
            "hypothesis_if_failed": "private_plane_index_semantics_mismatch",
        },
        {
            "name": "private_accumulators_do_not_populate_public_outputs",
            "passed": True,
            "expected": "private helpers return benchmark_flux_planes without public Result DFT/flux",
            "actual": "covered by private no-public-leak regression tests",
            "hypothesis_if_failed": "private_reference_helper_public_surface_leak",
        },
    ]


def _private_interface_floor_investigation_metadata(
    *,
    fixture: _FluxFixtureConfig,
    baseline_snapshot: dict[str, object],
    baseline_metrics: dict[str, float | int],
    dominant_reference_quality_blocker: str,
) -> dict[str, object]:
    thresholds_checksum = _reference_quality_thresholds_checksum()
    baseline_packet_id = f"{fixture.name}:interface_floor:{thresholds_checksum[:12]}"
    slow_command = _PRIVATE_TRUE_RT_SLOW_COMMAND
    scored_command = {"slow_command_required": True, "slow_command": slow_command}
    baseline_candidate = _interface_distance_candidate_record(
        candidate_id="baseline_boundary_expanded",
        role="current_baseline",
        fixture=fixture,
        snapshot=baseline_snapshot,
        baseline_metrics=baseline_metrics,
        dominant_metric=dominant_reference_quality_blocker,
        baseline_packet_id=baseline_packet_id,
        command_requirement={"slow_command_required": False, "slow_command": None},
    )
    nearer_snapshot = _private_tfsf_reference_quality_snapshot(fixture=_FluxFixture)
    nearer_candidate = _interface_distance_candidate_record(
        candidate_id="nearer_current_bounded_control",
        role="nearer_interface_control",
        fixture=_FluxFixture,
        snapshot=nearer_snapshot,
        baseline_metrics=baseline_metrics,
        dominant_metric=dominant_reference_quality_blocker,
        baseline_packet_id=baseline_packet_id,
        command_requirement=scored_command,
    )
    skipped_larger_candidate = {
        "candidate_id": "larger_fine_guard_same_source_planes",
        "role": "larger_guard_control",
        "status": "skipped",
        "predeclared": True,
        "reason": (
            "current boundary_expanded fixture already supersedes the nearer "
            "current_bounded geometry; an even larger slow fixture is deferred "
            "to the follow-up repair plan to avoid broadening this diagnostic pass"
        ),
        "command_requirement": scored_command,
    }
    interface_distance_candidates = [
        baseline_candidate,
        nearer_candidate,
        skipped_larger_candidate,
    ]
    energy_transfer = _interface_energy_transfer_summary(
        [
            candidate
            for candidate in interface_distance_candidates
            if candidate.get("status") == "scored"
        ]
    )
    direct_invariant_tests = _private_direct_invariant_tests()
    failed_invariant = next(
        (test for test in direct_invariant_tests if test["passed"] is False),
        None,
    )
    hook_justification = _default_hook_contingency_justification(
        per_rung_status={
            "rung1": "negative",
            "rung2": "negative",
            "rung3": "negative",
            "rung4": "negative",
            "rung5": "positive",
        },
        direct_invariant_test=failed_invariant,
        fixed_candidate=baseline_packet_id,
    )

    if failed_invariant is not None:
        subclass = "hook_contingency_direct_invariant"
        next_prerequisite = (
            "open a private hook-order/sign/index plan with dedicated "
            "jit_runner.py and sbp_sat_3d.py regressions"
        )
    elif bool(energy_transfer["interface_residual_stable"]) and bool(
        energy_transfer["uniform_reference_below_threshold"]
    ):
        subclass = "coarse_fine_energy_transfer_mismatch"
        next_prerequisite = _INTERFACE_FLOOR_NEXT_PREREQUISITE
    else:
        subclass = "unresolved_interface_floor"
        next_prerequisite = (
            "add a larger private interface-distance or CPML-proximity control "
            "before low-level hook experiments"
        )

    cpml_proximity_controls = [
        {
            "candidate_id": "larger_domain_or_absorber_guard_control",
            "status": "not_entered",
            "predeclared": True,
            "reason": (
                "private front/back energy-transfer residual was already stable "
                "across the scored boundary-expanded and nearer bounded "
                "interface candidates"
            )
            if subclass == "coarse_fine_energy_transfer_mismatch"
            else "deferred because current interface-distance evidence did not justify a CPML-specific conclusion",
        },
        {
            "candidate_id": "mixed_periodic_cpml_shortcut",
            "status": "blocked_by_support_contract",
            "predeclared": True,
            "reason": "mixed periodic+CPML remains unsupported in the SBP-SAT lane",
        },
    ]

    return {
        "interface_floor_investigation_status": "complete",
        "interface_floor_subclass": subclass,
        "interface_distance_sensitivity": "persistent"
        if subclass == "coarse_fine_energy_transfer_mismatch"
        else "unresolved",
        "interface_distance_candidates": interface_distance_candidates,
        "interface_energy_transfer_diagnostics": energy_transfer,
        "cpml_proximity_controls": cpml_proximity_controls,
        "direct_invariant_tests": direct_invariant_tests,
        "hook_contingency_justification": hook_justification,
        "interface_floor_next_prerequisite": next_prerequisite,
        "follow_up_recommendation": next_prerequisite,
    }


def _private_energy_transfer_repair_metadata(
    *,
    fixture: _FluxFixtureConfig,
    baseline_snapshot: dict[str, object],
    baseline_metrics: dict[str, float | int],
) -> dict[str, object]:
    """Score bounded private repair candidates without public promotion."""

    baseline_residual = _front_back_ratio_residual(
        uniform_signed_flux=baseline_snapshot["ref_run"].signed_flux,
        subgrid_signed_flux=baseline_snapshot["run"].signed_flux,
        mask=baseline_snapshot["freq_mask"],
        fixture=fixture,
    )
    slow_command = _PRIVATE_TRUE_RT_SLOW_COMMAND
    candidates: list[dict[str, object]] = []
    for tau in _PRIVATE_REPAIR_TAU_CANDIDATES:
        candidate_fixture = _fixture_with_tau(fixture, tau)
        if np.isclose(tau, fixture.tau):
            snapshot = baseline_snapshot
            command_requirement = {
                "slow_command_required": False,
                "slow_command": None,
            }
        else:
            snapshot = _private_tfsf_subgrid_quality_snapshot(
                fixture=candidate_fixture,
                ref_run=baseline_snapshot["ref_run"],
                source_eta0_consistency=baseline_snapshot["source_eta0_consistency"],
            )
            command_requirement = {
                "slow_command_required": True,
                "slow_command": slow_command,
            }
        tau_label = str(tau).replace(".", "p")
        candidates.append(
            _private_repair_candidate_record(
                candidate_id=f"tau_sensitivity_{tau_label}",
                hypothesis="tau_sensitivity_private_fixture_only",
                fixture=candidate_fixture,
                snapshot=snapshot,
                baseline_metrics=baseline_metrics,
                baseline_residual=baseline_residual,
                command_requirement=command_requirement,
            )
        )

    outcome = _private_repair_outcome_from_candidates(candidates)
    status = outcome["private_energy_transfer_repair_status"]
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate["candidate_id"] == outcome["selected_repair_candidate_id"]
        ),
        None,
    )
    low_level_status = (
        "not_entered_no_tau_candidate_met_material_gate"
        if status == "no_material_repair"
        else "not_entered_tau_candidate_private_only"
    )
    repair_packet = {
        "status": status,
        "candidate_policy": (
            "tau sensitivity uses existing private fixture plumbing only; public "
            "tau default/API behavior remains unchanged"
        ),
        "tau_candidates": [float(tau) for tau in _PRIVATE_REPAIR_TAU_CANDIDATES],
        "baseline_max_ratio_error": float(baseline_residual["max_ratio_error"]),
        "baseline_energy_transfer": baseline_residual,
        "selected_repair_candidate_id": outcome["selected_repair_candidate_id"],
        "repair_candidate_id": outcome["selected_repair_candidate_id"],
        "repair_hypothesis": None
        if selected is None
        else selected["repair_hypothesis"],
        "candidate_max_ratio_error": None
        if selected is None
        else selected["candidate_max_ratio_error"],
        "relative_improvement": None
        if selected is None
        else selected["relative_improvement"],
        "paired_metric_regressions": []
        if selected is None
        else selected["paired_metric_regressions"],
        "accepted_private_repair": bool(outcome["accepted_private_repair"]),
        "candidates": candidates,
        "public_claim_allowed": False,
        "promotion_candidate_ready": False,
        "public_default_tau_changed": False,
        "public_api_behavior_changed": False,
        "kernel_edit_applied": False,
        "low_level_repair_status": low_level_status,
        "evidence_artifact_template": _PRIVATE_REPAIR_EVIDENCE_ARTIFACT_TEMPLATE,
        "baseline_artifact_required": True,
        "pre_post_evidence_required": True,
        "next_prerequisite": outcome["next_prerequisite"],
        "outcome_reason": outcome["outcome_reason"],
    }
    return {
        "private_energy_transfer_repair_status": status,
        "private_energy_transfer_repair": repair_packet,
        "private_energy_transfer_repair_next_prerequisite": outcome[
            "next_prerequisite"
        ],
    }


def _empty_rt_gates() -> dict[str, bool]:
    return {
        "r_magnitude": False,
        "t_magnitude": False,
        "phase": False,
        "energy": False,
        "plane_shift_r": False,
        "plane_shift_t": False,
        "plane_shift_phase": False,
    }


@lru_cache(maxsize=None)
def _private_tfsf_incident_metadata() -> dict[str, object]:
    fixture = _BoundaryExpandedFluxFixture
    ref_run = _run_flux_fixture(
        subgrid=False,
        slab=False,
        fixture=fixture,
        source_kind="private_tfsf",
    )
    run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=fixture,
        source_kind="private_tfsf",
    )
    if not np.allclose(ref_run.dt, run.dt):
        return {
            "classification": "fail",
            "reason": "private TFSF reference/subgrid timestep mismatch",
            "source_contract": "private_tfsf_style_incident",
            "normalization": "same_contract_private_reference_vacuum_gated",
            "reference_missing": False,
            "reference_quality_ready": False,
            "fixture_quality_ready": False,
            "slab_rt_scored": False,
            "public_claim_allowed": False,
        }
    for label, array in {
        "reference_front_complex": ref_run.complex_flux[0],
        "reference_back_complex": ref_run.complex_flux[1],
        "reference_front_signed": ref_run.signed_flux[0],
        "reference_back_signed": ref_run.signed_flux[1],
        "front_complex": run.complex_flux[0],
        "back_complex": run.complex_flux[1],
        "front_signed": run.signed_flux[0],
        "back_signed": run.signed_flux[1],
    }.items():
        fail = _finite_or_fail(label, array)
        if fail is not None:
            return fail | {
                "source_contract": "private_tfsf_style_incident",
                "normalization": "same_contract_private_reference_vacuum_gated",
                "reference_missing": False,
                "reference_quality_ready": False,
                "fixture_quality_ready": False,
                "slab_rt_scored": False,
                "public_claim_allowed": False,
            }

    freq_mask = _claims_bearing_passband(run.complex_flux, run.signed_flux)
    uniformity = _transverse_uniformity_metadata(
        run.planes,
        freq_mask,
        fixture,
        component="ex",
    )
    front_signed = np.asarray(run.signed_flux[0])
    back_signed = np.asarray(run.signed_flux[1])
    scored_front = front_signed[freq_mask]
    scored_back = back_signed[freq_mask]
    usable_passband = int(np.sum(freq_mask)) >= _MIN_CLAIMS_BEARING_BINS
    positive_z_flux = bool(
        usable_passband and np.all(scored_front > 0.0) and np.all(scored_back > 0.0)
    )
    nonfloor_flux = bool(
        usable_passband
        and np.all(np.abs(scored_front) >= _NORMALIZATION_FLOOR)
        and np.all(np.abs(scored_back) >= _NORMALIZATION_FLOOR)
    )
    incident_quality_gates = {
        "usable_passband": bool(usable_passband),
        "transverse_uniformity": bool(uniformity["passed"]),
        "analytic_incident_consistency": bool(nonfloor_flux),
    }
    vacuum_stability = _vacuum_stability_metadata(
        ref_run.complex_flux,
        run.complex_flux,
        freq_mask,
    )
    source_eta0_consistency = _private_tfsf_source_eta0_metadata(fixture)
    baseline_metrics = _reference_quality_metrics(
        usable_bins=int(np.sum(freq_mask)),
        uniformity=uniformity,
        vacuum_stability=vacuum_stability,
        source_eta0_consistency=source_eta0_consistency,
    )
    baseline_snapshot = {
        "ref_run": ref_run,
        "run": run,
        "freq_mask": freq_mask,
        "uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "source_eta0_consistency": source_eta0_consistency,
        "metrics": baseline_metrics,
        "usable_bins": int(np.sum(freq_mask)),
    }
    reference_quality_blockers = _reference_quality_blocker_ranking(
        usable_bins=int(np.sum(freq_mask)),
        nonfloor_flux=nonfloor_flux,
        uniformity=uniformity,
        vacuum_stability=vacuum_stability,
    )
    dominant_reference_quality_blocker = _dominant_reference_quality_blocker(
        reference_quality_blockers
    )
    reference_quality_ready = bool(
        incident_quality_gates["usable_passband"]
        and incident_quality_gates["transverse_uniformity"]
        and incident_quality_gates["analytic_incident_consistency"]
        and vacuum_stability["passed"]
    )
    fixture_quality_gates = {
        **incident_quality_gates,
        "same_contract_reference": True,
        "vacuum_stability": bool(vacuum_stability["passed"]),
    }
    base_metadata: dict[str, object] = {
        "fixture": ("boundary_expanded_private_tfsf_style_incident_flux_plane_vacuum"),
        "fixture_name": fixture.name,
        "fixture_parameters": fixture.to_metadata(),
        "source_contract": "private_tfsf_style_incident",
        "normalization": "same_contract_private_reference_vacuum_gated",
        "reference_missing": False,
        "reference_contract": "private_tfsf_style_uniform_reference",
        "reference_quality_ready": reference_quality_ready,
        "public_claim_allowed": False,
        "usable_bins": int(np.sum(freq_mask)),
        "scored_freqs_hz": fixture.scored_freqs[freq_mask].tolist(),
        "fixture_quality_gates": fixture_quality_gates,
        "reference_quality_thresholds": _reference_quality_thresholds(),
        "reference_quality_blockers": reference_quality_blockers,
        "dominant_reference_quality_blocker": dominant_reference_quality_blocker,
        "predeclared_candidate_policy": (
            "baseline full-aperture fixture remains reported; any future "
            "central-core or source-contract candidate must be predeclared in "
            "metadata before slow scoring and must keep these same thresholds"
        ),
        "positive_z_flux": positive_z_flux,
        "transverse_uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "source_eta0_consistency": source_eta0_consistency,
        "reference_vacuum_flux_diagnostics": _flux_diagnostics(
            ref_run.complex_flux,
            ref_run.signed_flux,
            fixture,
        ),
        "subgrid_vacuum_flux_diagnostics": _flux_diagnostics(
            run.complex_flux,
            run.signed_flux,
            fixture,
        ),
        "no_go_reason": _TFSF_NO_GO_REASON,
        "diagnostic_basis": (
            "This private fixture injects +z ex/hy incident fields through "
            "private post-H/post-E slots and accumulates only private "
            "fine-owned flux/DFT planes. A same-contract private uniform "
            "reference now exists, but it remains benchmark-only, not public "
            "TFSF, and does not expose public flux, DFT, S-parameter, port, "
            "or true R/T observables."
        ),
    }
    base_metadata.update(
        _private_tfsf_causal_ladder_metadata(
            fixture=fixture,
            baseline_metrics=baseline_metrics,
            dominant_reference_quality_blocker=dominant_reference_quality_blocker,
            source_eta0_consistency=source_eta0_consistency,
        )
    )
    base_metadata.update(
        _private_interface_floor_investigation_metadata(
            fixture=fixture,
            baseline_snapshot=baseline_snapshot,
            baseline_metrics=baseline_metrics,
            dominant_reference_quality_blocker=dominant_reference_quality_blocker,
        )
    )
    base_metadata.update(
        _private_energy_transfer_repair_metadata(
            fixture=fixture,
            baseline_snapshot=baseline_snapshot,
            baseline_metrics=baseline_metrics,
        )
    )
    recovery_metadata = _private_time_centered_helper_fixture_quality_recovery_metadata(
        baseline_snapshot=baseline_snapshot
    )
    base_metadata.update(
        {
            "private_time_centered_helper_fixture_quality_recovery_status": (
                recovery_metadata["status"]
            ),
            "private_time_centered_helper_fixture_quality_recovery": (
                recovery_metadata
            ),
            "private_time_centered_helper_fixture_quality_recovery_next_prerequisite": (
                recovery_metadata["next_prerequisite"]
            ),
        }
    )
    base_metadata["follow_up_recommendation"] = base_metadata[
        "private_time_centered_helper_fixture_quality_recovery_next_prerequisite"
    ]
    if not reference_quality_ready:
        return base_metadata | {
            "classification": "inconclusive",
            "reason": (
                "same-contract private TFSF-style reference is implemented, "
                "but private time-centered helper recovery candidates failed "
                "unchanged fixture-quality gates; slab R/T scoring is "
                "intentionally skipped"
            ),
            "slab_rt_scored": False,
            "fixture_quality_ready": False,
            "incident_fixture_quality_ready": bool(
                incident_quality_gates["usable_passband"]
                and incident_quality_gates["transverse_uniformity"]
                and incident_quality_gates["analytic_incident_consistency"]
            ),
            "gates": _empty_rt_gates(),
            "blocking_diagnostic": (
                "same-contract private reference helper is present, but "
                "private uniform/subgrid vacuum parity, transverse uniformity, "
                "or usable-passband gates remain below threshold; the causal "
                f"ladder now classifies the row-2 blocker as {base_metadata['causal_class']} "
                "because source, plane-shift, and central-core aperture "
                "controls did not satisfy the paired material-improvement "
                "rule; the interface-floor investigation now records "
                f"{base_metadata['interface_floor_subclass']}; slab R/T scoring "
                "and public promotion stay closed; the private energy-transfer "
                "repair stage records "
                f"{base_metadata['private_energy_transfer_repair_status']}; "
                "the private time-centered helper fixture-quality recovery stage "
                f"records {recovery_metadata['terminal_outcome']}"
            ),
            "next_prerequisite": base_metadata[
                "private_time_centered_helper_fixture_quality_recovery_next_prerequisite"
            ],
        }

    slab_run = _run_flux_fixture(
        subgrid=True,
        slab=True,
        fixture=fixture,
        source_kind="private_tfsf",
    )
    for label, array in {
        "slab_front_complex": slab_run.complex_flux[0],
        "slab_back_complex": slab_run.complex_flux[1],
        "slab_front_signed": slab_run.signed_flux[0],
        "slab_back_signed": slab_run.signed_flux[1],
    }.items():
        fail = _finite_or_fail(label, array)
        if fail is not None:
            return (
                base_metadata
                | fail
                | {
                    "slab_rt_scored": True,
                    "fixture_quality_ready": True,
                    "gates": _empty_rt_gates(),
                    "blocking_diagnostic": (
                        "slab scoring was attempted after reference gates passed, "
                        "but slab flux output was non-finite"
                    ),
                    "next_prerequisite": (
                        "repair private slab normalization/scorer before public "
                        "promotion is reconsidered"
                    ),
                }
            )
    energy_balance = _energy_residual(
        slab_run.signed_flux,
        ref_run.signed_flux,
        freq_mask,
    )
    rt_gates = {
        **_empty_rt_gates(),
        "energy": bool(float(np.max(energy_balance)) <= 0.05),
    }
    classification = "pass" if all(rt_gates.values()) else "inconclusive"
    return {
        **base_metadata,
        "classification": classification,
        "reason": (
            "same-contract private reference passed, but private slab R/T gates "
            "remain below threshold"
            if classification == "inconclusive"
            else "same-contract private reference and slab gates passed internally"
        ),
        "slab_rt_scored": True,
        "fixture_quality_ready": True,
        "incident_fixture_quality_ready": True,
        "gates": rt_gates,
        "energy_balance_residual": float(np.max(energy_balance)),
        "subgrid_slab_flux_diagnostics": _flux_diagnostics(
            slab_run.complex_flux,
            slab_run.signed_flux,
            fixture,
        ),
        "blocking_diagnostic": (
            "slab scoring was attempted after reference gates passed, but one "
            "or more private slab gates remains below threshold"
        ),
        "next_prerequisite": (
            "repair private slab normalization/scorer before public promotion "
            "is reconsidered"
        ),
    }


@lru_cache(maxsize=None)
def _plane_rt_metadata() -> dict[str, object]:
    return _private_tfsf_incident_metadata()


@lru_cache(maxsize=None)
def _analytic_sheet_plane_rt_metadata() -> dict[str, object]:
    sweep = _boundary_expansion_sweep_metadata()
    selected = next(
        fixture
        for fixture in _RECOVERY_SWEEP_FIXTURES
        if fixture.name == sweep["selected_fixture"]
    )
    selected_quality = next(
        candidate
        for candidate in sweep["candidates"]
        if candidate["fixture_name"] == selected.name
    )
    cheap_quality_passed = all(selected_quality["fixture_quality_gates"].values())
    if not cheap_quality_passed:
        return {
            "classification": "inconclusive",
            "reason": "boundary-expanded analytic-sheet sweep did not recover fixture quality",
            "fixture": selected.fixture_key,
            "fixture_name": selected.name,
            "fixture_parameters": selected.to_metadata(),
            "source_contract": "private_analytic_sheet_source",
            "normalization": "vacuum_device_two_run_incident_normalized",
            "public_claim_allowed": False,
            "boundary_expansion_sweep": sweep,
            "usable_bins": int(selected_quality["usable_bins"]),
            "scored_freqs_hz": selected_quality["scored_freqs_hz"],
            "fixture_quality_gates": {
                **selected_quality["fixture_quality_gates"],
                "plane_location": False,
            },
            "gates": _empty_rt_gates(),
            "transverse_uniformity": selected_quality["transverse_uniformity"],
            "vacuum_stability": selected_quality["vacuum_stability"],
            "no_go_reason": _NO_GO_REASON,
            "blocking_diagnostic": (
                "the bounded geometry sweep did not produce a canonical "
                "private sheet fixture with passing usable-passband, "
                "transverse-uniformity, and vacuum-stability gates; full "
                "incident-normalized R/T scoring is intentionally skipped"
            ),
            "next_prerequisite": _NEXT_PREREQUISITE,
            "diagnostic_basis": (
                "Candidate selection now uses a bounded boundary-expanded "
                "analytic-sheet sweep before full R/T scoring. Public support "
                "remains deferred and unsupported public observables stay "
                "hard-failing."
            ),
        }

    ref_vac_run = _run_flux_fixture(subgrid=False, slab=False, fixture=selected)
    ref_slab_run = _run_flux_fixture(subgrid=False, slab=True, fixture=selected)
    sub_vac_run = _run_flux_fixture(subgrid=True, slab=False, fixture=selected)
    sub_slab_run = _run_flux_fixture(subgrid=True, slab=True, fixture=selected)
    shift_vac_run = _run_flux_fixture(
        subgrid=True,
        slab=False,
        fixture=selected,
        plane_shift_cells=1,
    )
    shift_slab_run = _run_flux_fixture(
        subgrid=True,
        slab=True,
        fixture=selected,
        plane_shift_cells=1,
    )
    dt_ref, vac_ref, signed_vac_ref = (
        ref_vac_run.dt,
        ref_vac_run.complex_flux,
        ref_vac_run.signed_flux,
    )
    dt_ref_slab, slab_ref, signed_ref_slab = (
        ref_slab_run.dt,
        ref_slab_run.complex_flux,
        ref_slab_run.signed_flux,
    )
    dt_sub, vac_sub, signed_vac_sub = (
        sub_vac_run.dt,
        sub_vac_run.complex_flux,
        sub_vac_run.signed_flux,
    )
    dt_sub_slab, slab_sub, signed_sub_slab = (
        sub_slab_run.dt,
        sub_slab_run.complex_flux,
        sub_slab_run.signed_flux,
    )
    dt_shift_vac, vac_shift = shift_vac_run.dt, shift_vac_run.complex_flux
    dt_shift_slab, slab_shift = shift_slab_run.dt, shift_slab_run.complex_flux

    if not np.allclose(
        [dt_ref_slab, dt_sub, dt_sub_slab, dt_shift_vac, dt_shift_slab],
        dt_ref,
    ):
        return {
            "classification": "fail",
            "reason": "fixture timesteps are inconsistent",
        }

    for label, arrays in {
        "vac_ref": vac_ref,
        "slab_ref": slab_ref,
        "vac_sub": vac_sub,
        "slab_sub": slab_sub,
        "vac_shift": vac_shift,
        "slab_shift": slab_shift,
    }.items():
        for idx, array in enumerate(arrays):
            fail = _finite_or_fail(f"{label}_{idx}", array)
            if fail is not None:
                return fail

    freq_mask = _claims_bearing_passband(vac_sub, signed_vac_sub)
    uniformity = _transverse_uniformity_metadata(
        sub_vac_run.planes,
        freq_mask,
        selected,
    )
    vacuum_stability = _vacuum_stability_metadata(vac_ref, vac_sub, freq_mask)
    if int(np.sum(freq_mask)) < _MIN_CLAIMS_BEARING_BINS:
        return {
            "classification": "inconclusive",
            "reason": "no claims-bearing non-floor passband bins",
            "fixture": selected.fixture_key,
            "fixture_name": selected.name,
            "fixture_parameters": selected.to_metadata(),
            "source_contract": "private_analytic_sheet_source",
            "normalization": "vacuum_device_two_run_incident_normalized",
            "public_claim_allowed": False,
            "boundary_expansion_sweep": sweep,
            "usable_bins": int(np.sum(freq_mask)),
            "front_abs": np.abs(vac_sub[0]).tolist(),
            "back_abs": np.abs(vac_sub[1]).tolist(),
            "fixture_quality_gates": {
                "usable_passband": False,
                "transverse_uniformity": bool(uniformity["passed"]),
                "vacuum_stability": bool(vacuum_stability["passed"]),
                "plane_location": False,
            },
            "gates": _empty_rt_gates(),
            "transverse_uniformity": uniformity,
            "vacuum_stability": vacuum_stability,
            "no_go_reason": _NO_GO_REASON,
            "blocking_diagnostic": (
                "the private analytic sheet did not produce at least two "
                "vacuum front/back bins above both the 20% peak and "
                "1e12×normalization-floor thresholds"
            ),
            "next_prerequisite": _NEXT_PREREQUISITE,
        }

    def rt(vac: tuple[np.ndarray, np.ndarray], slab: tuple[np.ndarray, np.ndarray]):
        inc_front = np.where(
            np.abs(vac[0]) >= _NORMALIZATION_FLOOR,
            vac[0],
            _NORMALIZATION_FLOOR + 0j,
        )
        inc_back = np.where(
            np.abs(vac[1]) >= _NORMALIZATION_FLOOR,
            vac[1],
            _NORMALIZATION_FLOOR + 0j,
        )
        return {
            "R": (slab[0] - vac[0]) / inc_front,
            "T": slab[1] / inc_back,
        }

    rt_ref = rt(vac_ref, slab_ref)
    rt_sub = rt(vac_sub, slab_sub)
    rt_shift = rt(vac_shift, slab_shift)
    r_ref = rt_ref["R"][freq_mask]
    t_ref = rt_ref["T"][freq_mask]
    r_sub = rt_sub["R"][freq_mask]
    t_sub = rt_sub["T"][freq_mask]
    r_shift = rt_shift["R"][freq_mask]
    t_shift = rt_shift["T"][freq_mask]

    r_mag_error = _relative_magnitude_error(r_sub, r_ref)
    t_mag_error = _relative_magnitude_error(t_sub, t_ref)
    phase_refs = np.concatenate(
        [r_ref[np.abs(r_ref) >= 0.05], t_ref[np.abs(t_ref) >= 0.05]]
    )
    phase_tests = np.concatenate(
        [r_sub[np.abs(r_ref) >= 0.05], t_sub[np.abs(t_ref) >= 0.05]]
    )
    phase_error = (
        _phase_error_deg(phase_tests, phase_refs)
        if len(phase_refs)
        else np.array([0.0], dtype=np.float64)
    )
    r_shift_delta = np.abs(np.abs(r_shift) - np.abs(r_sub)) / np.maximum(
        np.abs(r_sub),
        _NORMALIZATION_FLOOR,
    )
    t_shift_delta = np.abs(np.abs(t_shift) - np.abs(t_sub)) / np.maximum(
        np.abs(t_sub),
        _NORMALIZATION_FLOOR,
    )
    shift_phase_refs = np.concatenate(
        [
            r_sub[np.abs(r_sub) >= 0.05],
            t_sub[np.abs(t_sub) >= 0.05],
        ]
    )
    shift_phase_tests = np.concatenate(
        [
            r_shift[np.abs(r_sub) >= 0.05],
            t_shift[np.abs(t_sub) >= 0.05],
        ]
    )
    shift_phase_error = (
        _phase_error_deg(shift_phase_tests, shift_phase_refs)
        if len(shift_phase_refs)
        else np.array([0.0], dtype=np.float64)
    )
    # Signed plane-flux energy is advisory for this private gate.  It is used
    # to detect gross normalization drift, not to promote public true-R/T.
    energy_balance = _energy_residual(signed_sub_slab, signed_vac_sub, freq_mask)
    uniform_energy_balance = _energy_residual(
        signed_ref_slab,
        signed_vac_ref,
        freq_mask,
    )
    energy_delta = np.abs(energy_balance - uniform_energy_balance)
    unfloored_energy_balance = _unfloored_energy_residual(
        signed_sub_slab,
        signed_vac_sub,
        freq_mask,
    )

    rt_gates = {
        "r_magnitude": float(np.max(r_mag_error)) <= 0.05,
        "t_magnitude": float(np.max(t_mag_error)) <= 0.05,
        "phase": float(np.max(phase_error)) <= 5.0,
        "energy": float(np.max(energy_balance)) <= 0.05,
        "plane_shift_r": float(np.max(r_shift_delta)) <= 0.05,
        "plane_shift_t": float(np.max(t_shift_delta)) <= 0.05,
        "plane_shift_phase": float(np.max(shift_phase_error)) <= 5.0,
    }
    fixture_quality_gates = {
        "usable_passband": int(np.sum(freq_mask)) >= _MIN_CLAIMS_BEARING_BINS,
        "transverse_uniformity": bool(uniformity["passed"]),
        "vacuum_stability": bool(vacuum_stability["passed"]),
        "plane_location": bool(
            rt_gates["plane_shift_r"]
            and rt_gates["plane_shift_t"]
            and rt_gates["plane_shift_phase"]
        ),
    }
    classification = (
        "pass"
        if all(fixture_quality_gates.values()) and all(rt_gates.values())
        else "inconclusive"
    )
    return {
        "classification": classification,
        "gates": rt_gates,
        "fixture_quality_gates": fixture_quality_gates,
        "fixture": selected.fixture_key,
        "fixture_name": selected.name,
        "fixture_parameters": selected.to_metadata(),
        "source_contract": "private_analytic_sheet_source",
        "normalization": "vacuum_device_two_run_incident_normalized",
        "boundary_expansion_sweep": sweep,
        "scored_freqs_hz": selected.scored_freqs[freq_mask].tolist(),
        "usable_bins": int(np.sum(freq_mask)),
        "max_r_magnitude_error": float(np.max(r_mag_error)),
        "max_t_magnitude_error": float(np.max(t_mag_error)),
        "max_phase_error_deg": float(np.max(phase_error)),
        "energy_balance_residual": float(np.max(energy_balance)),
        "uniform_energy_balance_residual": float(np.max(uniform_energy_balance)),
        "energy_residual_delta_vs_uniform": float(np.max(energy_delta)),
        "unfloored_energy_balance_residual": float(np.nanmax(unfloored_energy_balance)),
        "normalization_floor": _NORMALIZATION_FLOOR,
        "public_claim_allowed": False,
        "transverse_uniformity": uniformity,
        "vacuum_stability": vacuum_stability,
        "usable_passband_threshold": {
            "min_bins": _MIN_CLAIMS_BEARING_BINS,
            "front_back_peak_fraction": 0.20,
            "nonfloor_factor_times_normalization_floor": _NONFLOOR_FACTOR,
        },
        "transverse_uniformity_threshold": {
            "magnitude_cv_max": 0.01,
            "phase_spread_deg_max": 1.0,
        },
        "vacuum_stability_threshold": {
            "relative_magnitude_error_max": 0.02,
            "phase_error_deg_max": 2.0,
        },
        "plane_location_threshold": {
            "relative_magnitude_delta_max": 0.05,
            "phase_delta_deg_max": 5.0,
        },
        "max_plane_shift_r_delta": float(np.max(r_shift_delta)),
        "max_plane_shift_t_delta": float(np.max(t_shift_delta)),
        "max_plane_shift_phase_error_deg": float(np.max(shift_phase_error)),
        "subgrid_vacuum_flux_diagnostics": _flux_diagnostics(
            vac_sub,
            signed_vac_sub,
            selected,
        ),
        "subgrid_slab_flux_diagnostics": _flux_diagnostics(
            slab_sub,
            signed_sub_slab,
            selected,
        ),
        "no_go_reason": _NO_GO_REASON,
        "blocking_diagnostic": (
            "one or more private analytic-sheet fixture-quality or "
            "incident-normalized R/T gates remains below threshold, so the "
            "gate cannot be promoted as public true R/T evidence"
        ),
        "next_prerequisite": _NEXT_PREREQUISITE,
        "diagnostic_basis": (
            "The benchmark now uses a private analytic sheet, front/back "
            "private flux planes, and vacuum/device two-run normalization; "
            "public support remains deferred unless every fixture-quality "
            "and R/T gate passes."
        ),
        "replacement_metric_allowed": bool(float(np.max(energy_delta)) <= 0.02),
    }


def _print_metadata(title: str, metadata: dict[str, object]) -> None:
    print(f"\n{title}:")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _fail_or_xfail_inconclusive(metadata: dict[str, object], reason: str) -> None:
    if metadata["classification"] == "fail":
        pytest.fail(str(metadata))
    if metadata["classification"] == "inconclusive":
        pytest.xfail(reason)


def _synthetic_reference_metrics(**overrides: float | int) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "transverse_phase_spread_deg": 10.0,
        "transverse_magnitude_cv": 0.10,
        "vacuum_relative_magnitude_error": 0.20,
        "vacuum_phase_error_deg": 20.0,
        "source_eta0_relative_error": 0.0,
        "usable_bins": 3,
    }
    metrics.update(overrides)
    return metrics


def test_causal_class_truth_table_requires_positive_and_guard_evidence():
    truth_table = _causal_truth_table()

    assert set(_DISALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES) <= set(truth_table)
    for causal_class, contract in truth_table.items():
        assert contract["positive_evidence"]
        assert contract["guard_evidence"]
        assert bool(contract["same_run_repair_allowed"]) == _same_run_repair_allowed(
            causal_class
        )
        if causal_class != "unresolved_after_ladder":
            assert (
                "baseline_metrics" in contract["guard_evidence"]
                or causal_class == "measurement_or_reference_helper"
            )


def test_single_metric_improvement_is_inconclusive_not_causal():
    baseline = _synthetic_reference_metrics()
    candidate = _synthetic_reference_metrics(transverse_phase_spread_deg=4.0)

    decision = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        dominant_metric="transverse_phase_spread_deg",
    )

    assert decision["dominant"]["passed"] is True
    assert decision["paired_passed"] is False
    assert decision["passed"] is False
    assert decision["classification_decision"] == "inconclusive"


def test_non_predeclared_slow_candidate_cannot_set_causal_class_or_reference_ready():
    baseline = _synthetic_reference_metrics()
    candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=0.5,
        transverse_magnitude_cv=0.005,
    )
    decision = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        dominant_metric="transverse_phase_spread_deg",
    )
    record = _causal_ladder_candidate_record(
        candidate_id="unregistered_best_case",
        rung="rung3_plane_placement_phase_center",
        parameters={"plane_shift_cells": 99},
        cheap_rationale="synthetic unregistered candidate",
        baseline_packet_id="synthetic-baseline",
        before_metrics=baseline,
        after_metrics=candidate,
        classification_decision=decision,
        predeclared=False,
    )

    assert decision["passed"] is True
    assert (
        _causal_candidate_classification(
            candidate=record,
            intended_class="plane_placement_or_phase_center",
        )
        == "inconclusive"
    )
    no_claim = _row2_no_claim_for_causal_class("unresolved_after_ladder")
    assert no_claim["reference_quality_ready"] is False
    assert no_claim["fixture_quality_ready"] is False


def test_reduced_or_core_aperture_candidate_keeps_full_aperture_metrics_visible():
    baseline = _synthetic_reference_metrics()
    candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=0.5,
        transverse_magnitude_cv=0.005,
    )
    record = _causal_ladder_candidate_record(
        candidate_id="rung4_central_core_aperture",
        rung="rung4_aperture_source_edge",
        parameters={"aperture_size_m": 0.016, "full_aperture_metrics_visible": True},
        cheap_rationale="synthetic central-core diagnostic",
        baseline_packet_id="synthetic-baseline",
        before_metrics=baseline,
        after_metrics=candidate,
        classification_decision=_material_improvement_decision(
            baseline_metrics=baseline,
            candidate_metrics=candidate,
            dominant_metric="transverse_phase_spread_deg",
        ),
        full_aperture_metrics=baseline,
        candidate_class="finite_aperture_source_edge",
    )

    assert record["full_aperture_metrics"] == baseline
    assert record["parameters"]["full_aperture_metrics_visible"] is True


def test_material_improvement_rule_requires_dominant_blocker_or_threshold_crossing():
    baseline = _synthetic_reference_metrics()
    weak_candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=6.0,
        transverse_magnitude_cv=0.005,
    )
    threshold_candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=1.0,
        transverse_magnitude_cv=0.005,
    )

    weak = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=weak_candidate,
        dominant_metric="transverse_phase_spread_deg",
    )
    crossed = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=threshold_candidate,
        dominant_metric="transverse_phase_spread_deg",
    )

    assert weak["dominant"]["passed"] is False
    assert weak["passed"] is False
    assert crossed["dominant"]["threshold_crossed"] is True
    assert crossed["passed"] is True


def test_material_improvement_rule_requires_paired_metric_improvement():
    baseline = _synthetic_reference_metrics()
    candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=1.0,
        transverse_magnitude_cv=0.11,
        vacuum_phase_error_deg=21.0,
    )

    decision = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        dominant_metric="transverse_phase_spread_deg",
    )

    assert decision["dominant"]["passed"] is True
    assert decision["paired_passed"] is False
    assert decision["passed"] is False


def test_material_improvement_rule_rejects_usable_bin_regression():
    baseline = _synthetic_reference_metrics()
    candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=0.5,
        transverse_magnitude_cv=0.005,
        usable_bins=1,
    )

    decision = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        dominant_metric="transverse_phase_spread_deg",
    )

    assert decision["usable_bins_passed"] is False
    assert decision["passed"] is False


def test_material_improvement_rule_marks_new_blocker_regression_as_tradeoff():
    baseline = _synthetic_reference_metrics(vacuum_relative_magnitude_error=0.03)
    candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=0.5,
        transverse_magnitude_cv=0.005,
        vacuum_relative_magnitude_error=0.05,
    )

    decision = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        dominant_metric="transverse_phase_spread_deg",
    )

    assert decision["new_blocker_regressions"]
    assert decision["classification_decision"] == "tradeoff_inconclusive"
    assert decision["passed"] is False


def test_source_eta0_consistency_uses_existing_vacuum_magnitude_tolerance():
    electric = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    eta0 = 400.0
    magnetic_ok = electric / eta0 * 1.01
    magnetic_bad = electric / eta0 * 1.03

    assert _source_eta0_consistency_metadata(
        electric,
        magnetic_ok,
        eta0=eta0,
    ) == {
        "metric": "source_eta0_relative_error",
        "relative_error": pytest.approx(0.01),
        "threshold": _VACUUM_MAGNITUDE_ERROR_MAX,
        "passed": True,
        "formula": "max(abs(abs(H)-abs(E/eta0))/max(abs(E/eta0), floor))",
    }
    assert (
        _source_eta0_consistency_metadata(electric, magnetic_bad, eta0=eta0)["passed"]
        is False
    )


def test_causal_ladder_candidate_records_have_minimal_schema():
    baseline = _synthetic_reference_metrics()
    record = _causal_ladder_candidate_record(
        candidate_id="rung3_plane_shift_plus_2_cells",
        rung="rung3_plane_placement_phase_center",
        parameters={"plane_shift_cells": 2},
        cheap_rationale="synthetic candidate",
        baseline_packet_id="synthetic-baseline",
        before_metrics=baseline,
        after_metrics=baseline,
        classification_decision=_material_improvement_decision(
            baseline_metrics=baseline,
            candidate_metrics=baseline,
            dominant_metric="transverse_phase_spread_deg",
        ),
    )

    for key in (
        "candidate_id",
        "rung",
        "parameters",
        "cheap_rationale",
        "baseline_packet_id",
        "thresholds_checksum",
        "command_requirement",
        "before_metrics",
        "after_metrics",
        "classification_decision",
    ):
        assert key in record


def test_threshold_checksum_change_blocks_reference_quality_ready():
    baseline = _synthetic_reference_metrics()
    candidate = _synthetic_reference_metrics(
        transverse_phase_spread_deg=0.5,
        transverse_magnitude_cv=0.005,
    )
    changed_thresholds = _reference_quality_thresholds()
    changed_thresholds["transverse_uniformity"] = {
        **changed_thresholds["transverse_uniformity"],
        "phase_spread_deg_max": 5.0,
    }

    decision = _material_improvement_decision(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        dominant_metric="transverse_phase_spread_deg",
        thresholds_checksum=_reference_quality_thresholds_checksum(changed_thresholds),
    )

    assert decision["thresholds_checksum_matches"] is False
    assert decision["classification_decision"] == "threshold_mismatch_inconclusive"
    assert decision["passed"] is False


def test_hook_contingency_gate_defaults_ineligible_until_rungs_one_to_five_negative():
    default = _default_hook_contingency_justification(
        per_rung_status={
            "rung1": "negative",
            "rung2": "negative",
            "rung3": "negative",
            "rung4": "negative",
            "rung5": "positive",
        }
    )
    eligible = _default_hook_contingency_justification(
        per_rung_status={f"rung{i}": "negative" for i in range(1, 6)}
    )

    assert default["eligible"] is False
    assert eligible["eligible"] is True


def test_hook_contingency_justification_records_required_fields_when_eligible():
    hook = _default_hook_contingency_justification(
        direct_invariant_test={
            "name": "synthetic_index_semantics",
            "passed": False,
            "expected": "H samples index-1",
            "actual": "H samples index",
        }
    )

    assert hook["eligible"] is True
    for key in (
        "per_rung_status",
        "direct_invariant_test",
        "expected",
        "actual",
        "fixed_candidate",
        "hypothesis",
        "why_hook_scope_is_allowed",
        "rollback_policy",
    ):
        assert key in hook


def test_same_run_repair_policy_allows_only_fixture_level_causal_classes():
    for causal_class in _ALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES:
        assert _same_run_repair_allowed(causal_class) is True
    for causal_class in _DISALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES:
        assert _same_run_repair_allowed(causal_class) is False


def test_disallowed_same_run_repair_classes_preserve_row2_no_claim_status():
    for causal_class in _DISALLOWED_SAME_RUN_REPAIR_CAUSAL_CLASSES:
        no_claim = _row2_no_claim_for_causal_class(causal_class)
        assert no_claim["reference_quality_ready"] is False
        assert no_claim["fixture_quality_ready"] is False
        assert no_claim["slab_rt_scored"] is False
        assert no_claim["public_claim_allowed"] is False
        assert no_claim["follow_up_recommendation"]


def test_interface_energy_transfer_residual_uses_front_back_ratio_formula():
    residual = _front_back_ratio_residual(
        uniform_signed_flux=(
            np.asarray([2.0, 4.0], dtype=np.float64),
            np.asarray([1.0, 2.0], dtype=np.float64),
        ),
        subgrid_signed_flux=(
            np.asarray([2.0, 4.0], dtype=np.float64),
            np.asarray([1.5, 3.0], dtype=np.float64),
        ),
        mask=np.asarray([True, True]),
        fixture=_FluxFixtureConfig(
            name="synthetic_ratio",
            fixture_key="synthetic_ratio",
            scored_freqs_tuple=(1.0, 2.0),
        ),
    )

    assert residual["front_back_ratio_formula"] == _FRONT_BACK_RATIO_FORMULA
    assert residual["ratio_error_formula"] == _FRONT_BACK_RATIO_ERROR_FORMULA
    assert residual["uniform_front_back_ratio"] == [0.5, 0.5]
    assert residual["subgrid_front_back_ratio"] == [0.75, 0.75]
    assert residual["ratio_error_by_freq"] == pytest.approx([0.5, 0.5])
    assert residual["max_ratio_error"] == pytest.approx(0.5)
    assert residual["front_back_ratio_sign_consistent"] is True


def test_interface_energy_transfer_summary_requires_stable_predeclared_residuals():
    candidates = [
        {
            "candidate_id": "baseline_boundary_expanded",
            "status": "scored",
            "energy_transfer": {
                "valid": True,
                "max_ratio_error": 0.10,
                "front_back_ratio_sign_consistent": True,
            },
        },
        {
            "candidate_id": "nearer_current_bounded_control",
            "status": "scored",
            "energy_transfer": {
                "valid": True,
                "max_ratio_error": 0.12,
                "front_back_ratio_sign_consistent": True,
            },
        },
    ]

    summary = _interface_energy_transfer_summary(candidates)

    assert summary["interface_residual_stable"] is True
    assert summary["stable_residual_evidence"] == (
        "stable_across_predeclared_candidates"
    )
    assert summary["uniform_reference_self_error"] == 0.0
    assert summary["uniform_reference_below_threshold"] is True


def test_private_tau_fixture_uses_existing_refinement_plumbing_only():
    candidate = _fixture_with_tau(_BoundaryExpandedFluxFixture, 0.75)

    assert _BoundaryExpandedFluxFixture.tau == 0.5
    assert _BoundaryExpandedFluxFixture.refinement["tau"] == 0.5
    assert candidate.tau == 0.75
    assert candidate.refinement["tau"] == 0.75
    assert candidate.name.endswith("tau_0p75")
    assert candidate.fixture_key.endswith("tau_0p75")


def test_private_repair_outcome_keeps_public_claim_handoff_closed():
    no_material = _private_repair_outcome_from_candidates(
        [
            {
                "status": "scored",
                "candidate_id": "tau_sensitivity_0p5",
                "candidate_max_ratio_error": 0.90,
                "material_improvement_passed": False,
                "strong_private_repair_passed": False,
            }
        ]
    )
    accepted = _private_repair_outcome_from_candidates(
        [
            {
                "status": "scored",
                "candidate_id": "tau_sensitivity_0p75",
                "candidate_max_ratio_error": 0.30,
                "material_improvement_passed": True,
                "strong_private_repair_passed": False,
            }
        ]
    )
    repaired = _private_repair_outcome_from_candidates(
        [
            {
                "status": "scored",
                "candidate_id": "tau_sensitivity_1p0",
                "candidate_max_ratio_error": 0.01,
                "material_improvement_passed": True,
                "strong_private_repair_passed": True,
            }
        ]
    )

    assert no_material["private_energy_transfer_repair_status"] == (
        "no_material_repair"
    )
    assert no_material["accepted_private_repair"] is False
    assert no_material["next_prerequisite"] == (
        _PRIVATE_REPAIR_NO_MATERIAL_NEXT_PREREQUISITE
    )
    assert accepted["private_energy_transfer_repair_status"] == (
        "accepted_private_candidate"
    )
    assert accepted["accepted_private_repair"] is True
    assert accepted["next_prerequisite"] == (_PRIVATE_REPAIR_ACCEPTED_NEXT_PREREQUISITE)
    assert repaired["private_energy_transfer_repair_status"] == (
        "repaired_private_floor"
    )
    assert repaired["accepted_private_repair"] is True
    assert repaired["next_prerequisite"] == (_PRIVATE_REPAIR_REPAIRED_NEXT_PREREQUISITE)


def test_interface_floor_direct_invariant_tests_keep_hook_closed_by_default():
    invariants = _private_direct_invariant_tests()

    assert invariants
    assert all(test["passed"] for test in invariants)
    hook = _default_hook_contingency_justification(
        per_rung_status={
            "rung1": "negative",
            "rung2": "negative",
            "rung3": "negative",
            "rung4": "negative",
            "rung5": "positive",
        },
        direct_invariant_test=None,
    )
    assert hook["eligible"] is False


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_flux_matches_uniform_reference_in_homogeneous_cpml_fixture():
    metadata = _homogeneous_parity_metadata()
    _print_metadata("SBP-SAT private flux homogeneous parity metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private analytic-sheet runtime parity is inconclusive for public "
        "promotion; support matrix must not promote public flux/DFT or true "
        "R/T.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_benchmark_vs_uniform_fine():
    metadata = _plane_rt_metadata()
    _print_metadata("SBP-SAT private TFSF-style fixture metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private TFSF-style incident fixture has a same-contract reference, "
        "but vacuum/reference gates are still inconclusive, so slab true R/T "
        "scoring and public promotion remain deferred.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_plane_shift_stability():
    metadata = _plane_rt_metadata()
    _print_metadata("SBP-SAT private TFSF-style no-shift metadata", metadata)
    _fail_or_xfail_inconclusive(
        metadata,
        "Private TFSF-style incident fixture has a same-contract reference, "
        "but fixture-quality gates remain inconclusive; do not run plane-shift "
        "slab R/T scoring or promote true R/T.",
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_private_plane_true_rt_no_go_metadata_is_explicit():
    metadata = _plane_rt_metadata()

    assert metadata["classification"] == "inconclusive"
    assert metadata["public_claim_allowed"] is False
    assert metadata["source_contract"] == "private_tfsf_style_incident"
    assert metadata["normalization"] == "same_contract_private_reference_vacuum_gated"
    assert metadata["reference_missing"] is False
    assert metadata["reference_quality_ready"] is False
    assert metadata["reference_contract"] == "private_tfsf_style_uniform_reference"
    assert metadata["slab_rt_scored"] is False
    assert metadata["fixture_quality_ready"] is False
    assert metadata["fixture_name"] == _BoundaryExpandedFluxFixture.name
    assert metadata["fixture"] == (
        "boundary_expanded_private_tfsf_style_incident_flux_plane_vacuum"
    )
    assert metadata["fixture_quality_gates"]["same_contract_reference"] is True
    assert metadata["fixture_quality_gates"]["vacuum_stability"] is False
    assert not all(metadata["fixture_quality_gates"].values())
    assert (
        metadata["dominant_reference_quality_blocker"] == "transverse_phase_spread_deg"
    )
    blockers = metadata["reference_quality_blockers"]
    assert blockers[0]["name"] == metadata["dominant_reference_quality_blocker"]
    assert blockers[0]["passed"] is False
    assert blockers[0]["severity_vs_threshold"] > 1.0
    assert metadata["reference_quality_thresholds"]["transverse_uniformity"] == {
        "magnitude_cv_max": _TRANSVERSE_MAGNITUDE_CV_MAX,
        "phase_spread_deg_max": _TRANSVERSE_PHASE_SPREAD_DEG_MAX,
    }
    assert "predeclared" in metadata["predeclared_candidate_policy"]
    assert metadata["causal_ladder_status"] in {
        "row2_causal_classified",
        "row2_unresolved_after_ladder",
    }
    assert metadata["causal_class"] in _causal_truth_table()
    assert metadata["material_improvement_rule"]["thresholds_checksum"] == (
        _reference_quality_thresholds_checksum()
    )
    assert metadata["hook_contingency_justification"]["eligible"] is False
    assert metadata["interface_floor_investigation_status"] == "complete"
    assert metadata["interface_floor_subclass"] in _INTERFACE_FLOOR_SUBCLASSES
    assert metadata["interface_floor_subclass"] == (
        "coarse_fine_energy_transfer_mismatch"
    )
    assert metadata["interface_distance_sensitivity"] == "persistent"
    assert len(metadata["interface_distance_candidates"]) >= 2
    assert {
        candidate["candidate_id"]
        for candidate in metadata["interface_distance_candidates"]
    } >= {"baseline_boundary_expanded", "nearer_current_bounded_control"}
    energy = metadata["interface_energy_transfer_diagnostics"]
    assert energy["front_back_ratio_formula"] == _FRONT_BACK_RATIO_FORMULA
    assert energy["interface_residual_stable"] is True
    assert energy["uniform_reference_below_threshold"] is True
    assert metadata["cpml_proximity_controls"]
    assert metadata["direct_invariant_tests"]
    assert all(test["passed"] for test in metadata["direct_invariant_tests"])
    repair = metadata["private_energy_transfer_repair"]
    assert metadata["private_energy_transfer_repair_status"] in _PRIVATE_REPAIR_STATUSES
    assert repair["status"] == metadata["private_energy_transfer_repair_status"]
    assert repair["public_claim_allowed"] is False
    assert repair["promotion_candidate_ready"] is False
    assert repair["public_default_tau_changed"] is False
    assert repair["public_api_behavior_changed"] is False
    assert repair["kernel_edit_applied"] is False
    assert repair["baseline_artifact_required"] is True
    assert repair["pre_post_evidence_required"] is True
    assert (
        repair["evidence_artifact_template"]
        == _PRIVATE_REPAIR_EVIDENCE_ARTIFACT_TEMPLATE
    )
    assert repair["candidate_policy"].startswith("tau sensitivity")
    assert repair["tau_candidates"] == list(_PRIVATE_REPAIR_TAU_CANDIDATES)
    assert len(repair["candidates"]) == len(_PRIVATE_REPAIR_TAU_CANDIDATES)
    assert repair["selected_repair_candidate_id"] in {
        candidate["candidate_id"] for candidate in repair["candidates"]
    }
    assert (
        repair["next_prerequisite"]
        == (metadata["private_energy_transfer_repair_next_prerequisite"])
    )
    recovery = metadata["private_time_centered_helper_fixture_quality_recovery"]
    assert metadata["private_time_centered_helper_fixture_quality_recovery_status"] == (
        "measurement_contract_or_interface_floor_persists"
    )
    assert (
        recovery["terminal_outcome"]
        == (metadata["private_time_centered_helper_fixture_quality_recovery_status"])
    )
    assert recovery["candidate_ladder_declared_before_slow_scoring"] is True
    assert recovery["candidate_count"] == 4
    assert recovery["thresholds_checksum"] == _reference_quality_thresholds_checksum()
    assert recovery["selected_candidate_id"] == "C0_current_helper_original_fixture"
    assert recovery["solver_hunk_touched"] is True
    assert recovery["solver_hunk_retained"] is False
    assert recovery["current_fixture_metrics_retained"] is True
    assert recovery["slab_rt_private_only"] is True
    assert recovery["slab_rt_public_claim_allowed"] is False
    assert recovery["fixture_quality_ready"] is False
    assert recovery["reference_quality_ready"] is False
    assert recovery["public_claim_allowed"] is False
    assert recovery["public_observable_promoted"] is False
    assert recovery["hook_experiment_allowed"] is False
    assert (
        recovery["next_prerequisite"]
        == metadata[
            "private_time_centered_helper_fixture_quality_recovery_next_prerequisite"
        ]
    )
    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in recovery["candidates"]
    }
    assert set(candidate_by_id) == {
        "C0_current_helper_original_fixture",
        "C1_center_core_measurement_control",
        "C2_one_cell_downstream_plane_control",
        "C3_helper_relaxation_0p05_original_fixture",
    }
    assert (
        candidate_by_id["C1_center_core_measurement_control"][
            "can_claim_original_fixture_recovery"
        ]
        is False
    )
    assert (
        candidate_by_id["C2_one_cell_downstream_plane_control"][
            "can_claim_original_fixture_recovery"
        ]
        is False
    )
    c3 = candidate_by_id["C3_helper_relaxation_0p05_original_fixture"]
    assert c3["solver_touch"] is True
    assert c3["rollback_required"] is True
    assert c3["rollback_verified"] is True
    assert c3["retained_solver_relaxation"] == 0.02
    assert (
        metadata["follow_up_recommendation"]
        == metadata[
            "private_time_centered_helper_fixture_quality_recovery_next_prerequisite"
        ]
    )
    assert metadata["causal_ladder_rungs"]["rung0_baseline_freeze"]["status"] == (
        "complete"
    )
    assert metadata["causal_ladder_candidates"]
    for candidate in metadata["causal_ladder_candidates"]:
        for key in (
            "candidate_id",
            "rung",
            "parameters",
            "cheap_rationale",
            "baseline_packet_id",
            "thresholds_checksum",
            "command_requirement",
            "before_metrics",
            "after_metrics",
            "classification_decision",
        ):
            assert key in candidate
    assert not any(metadata["gates"].values())
    assert metadata["no_go_reason"] == _TFSF_NO_GO_REASON
    assert (
        metadata["next_prerequisite"]
        == (
            metadata[
                "private_time_centered_helper_fixture_quality_recovery_next_prerequisite"
            ]
        )
    )
    assert (
        "same-contract private reference helper is present"
        in metadata["blocking_diagnostic"]
    )
    assert "coarse_fine_energy_transfer_mismatch" in metadata["blocking_diagnostic"]
    assert (
        metadata["private_energy_transfer_repair_status"]
        in metadata["blocking_diagnostic"]
    )
    assert (
        metadata["private_time_centered_helper_fixture_quality_recovery_status"]
        in metadata["blocking_diagnostic"]
    )
    assert "not public TFSF" in metadata["diagnostic_basis"]
