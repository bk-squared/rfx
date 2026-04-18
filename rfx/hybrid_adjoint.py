"""Hybrid adjoint helpers for the staged uniform time-series seam.

This module currently supports the narrow staged seam used by the hybrid
adjoint proof-of-concept and Phase 3A expansion work:

- uniform grid
- non-periodic
- lossless or Debye-dispersive materials with zero conductivity
- point-source / point-probe time-series objectives
- PEC boundaries
- CPML boundaries (including CPML + per-face PEC overrides)

Unsupported physics is expected to route back to the existing pure-AD
``Simulation.forward()`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import (
    EPS_0,
    MaterialArrays,
    UpdateCoeffs,
    FDTDState,
    init_state,
    precompute_coeffs,
    update_e,
    update_h,
    update_he_fast,
)
from rfx.boundaries.cpml import (
    CPMLAxisParams,
    CPMLState,
    apply_cpml_e,
    apply_cpml_h,
    init_cpml,
)
from rfx.boundaries.pec import apply_pec, apply_pec_faces
from rfx.grid import Grid
from rfx.materials.debye import DebyeCoeffs, DebyeState, init_debye, update_e_debye
from rfx.simulation import ProbeSpec, SourceSpec


class Phase1FieldState(NamedTuple):
    """Differentiable field-only carry for the Phase 1 replay seam."""

    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray


@dataclass(frozen=True)
class Phase1HybridInventory:
    """Carry/replay inventory for the extracted Phase 1 seam."""

    carry_fields: tuple[str, ...]
    carry_bytes: tuple[tuple[str, int], ...]
    total_carry_bytes: int
    replay_inputs: tuple[str, ...]
    replay_outputs: tuple[str, ...]


@dataclass(frozen=True)
class Phase1HybridPreparedRunnerState:
    """Seam-owned snapshot of the uniform runner-prepared Phase 1 state."""

    materials: MaterialArrays
    sources: list | tuple
    raw_phase1_sources: tuple[tuple[int, int, int, str, jnp.ndarray], ...]
    probes: list | tuple
    debye_spec: tuple | None
    debye: tuple | None
    lorentz: tuple | None
    ntff_box: object | None
    waveguide_ports: list | tuple | None
    periodic_bool: tuple[bool, bool, bool]
    cpml_axes_run: str
    pec_mask_local: jnp.ndarray | None
    pec_occupancy_local: jnp.ndarray | None
    pec_axes_run: str

    @classmethod
    def from_uniform_prepared(cls, prepared: object) -> Phase1HybridPreparedRunnerState:
        return cls(
            materials=prepared.materials,
            sources=prepared.sources,
            raw_phase1_sources=prepared.raw_phase1_sources,
            probes=prepared.probes,
            debye_spec=prepared.debye_spec,
            debye=prepared.debye,
            lorentz=prepared.lorentz,
            ntff_box=prepared.ntff_box,
            waveguide_ports=prepared.waveguide_ports,
            periodic_bool=prepared.periodic_bool,
            cpml_axes_run=prepared.cpml_axes_run,
            pec_mask_local=prepared.pec_mask_local,
            pec_occupancy_local=prepared.pec_occupancy_local,
            pec_axes_run=prepared.pec_axes_run,
        )


def phase1_forward_result(grid: Grid, time_series: jnp.ndarray) -> "ForwardResult":
    """Build the minimal ForwardResult for the Phase 1 seam."""

    from rfx.api import ForwardResult

    return ForwardResult(
        time_series=time_series,
        ntff_data=None,
        ntff_box=None,
        grid=grid,
        s_params=None,
        freqs=None,
    )


@dataclass(frozen=True)
class Phase1HybridContext:
    """Static replay context for the experimental Phase 1 hybrid lane."""

    grid: Grid
    boundary: str
    n_steps: int
    dt: float
    dx: float
    eps_r: jnp.ndarray
    mu_r: jnp.ndarray
    zero_sigma: jnp.ndarray
    debye_spec: tuple | None
    cpml_axes: str
    pec_axes: str
    pec_faces: tuple[str, ...]
    cpml_params: CPMLAxisParams | None
    src_waveforms_raw: jnp.ndarray
    src_meta: tuple[tuple[int, int, int, str], ...]
    prb_meta: tuple[tuple[int, int, int, str], ...]
    initial_state: Phase1FieldState
    inventory: Phase1HybridInventory

    @classmethod
    def from_inputs(cls, inputs: Phase1HybridInputs) -> Phase1HybridContext:
        assert inputs.grid is not None
        assert inputs.materials is not None
        assert inputs.n_steps is not None

        initial_fdtd = init_state(inputs.grid.shape, field_dtype=jnp.float32)
        initial_state = _from_fdtd(initial_fdtd)
        boundary = inputs.boundary
        if boundary not in {"pec", "cpml"}:
            raise ValueError(f"boundary={boundary!r} is unsupported")
        cpml_axes = inputs.cpml_axes or getattr(inputs.grid, "cpml_axes", "")
        pec_faces = inputs.pec_faces or tuple(sorted(getattr(inputs.grid, "pec_faces", set())))
        cpml_params = None
        src_waveforms_raw = (
            jnp.stack([waveform for _, _, _, _, waveform in inputs.raw_sources], axis=-1)
            if inputs.raw_sources
            else jnp.zeros((inputs.n_steps, 0), dtype=jnp.float32)
        )
        src_meta = tuple((i, j, k, component) for i, j, k, component, _ in inputs.raw_sources)
        prb_meta = tuple((prb.i, prb.j, prb.k, prb.component) for prb in inputs.probes)

        carry_fields = ("fdtd.ex", "fdtd.ey", "fdtd.ez", "fdtd.hx", "fdtd.hy", "fdtd.hz")
        carry_arrays: list[tuple[str, jnp.ndarray]] = list(zip(carry_fields, initial_state))
        replay_inputs = ["eps_r", "mu_r", "source_waveforms_raw"]
        replay_outputs = ["time_series", "final_fields"]
        debye_spec = inputs.debye_spec
        if debye_spec is not None:
            _, initial_debye_state = _runtime_debye(inputs.grid.dt, inputs.materials, debye_spec)
            carry_arrays.extend(
                (f"debye.{name}", arr)
                for name, arr in zip(initial_debye_state._fields, initial_debye_state)
            )
            replay_inputs.append("debye_spec")
            replay_outputs.append("final_debye_state")
        if boundary == "cpml":
            cpml_params, initial_cpml_state = init_cpml(inputs.grid, pec_faces=set(pec_faces))
            carry_arrays.extend(
                (f"cpml.{name}", arr)
                for name, arr in zip(initial_cpml_state._fields, initial_cpml_state)
            )
            replay_inputs.extend(("cpml_params", "pec_faces"))
            replay_outputs.append("final_cpml_state")
        carry_bytes = tuple(
            (name, int(arr.size * arr.dtype.itemsize))
            for name, arr in carry_arrays
        )
        inventory = Phase1HybridInventory(
            carry_fields=tuple(name for name, _ in carry_arrays),
            carry_bytes=carry_bytes,
            total_carry_bytes=sum(size for _, size in carry_bytes),
            replay_inputs=tuple(replay_inputs),
            replay_outputs=tuple(replay_outputs),
        )
        return cls(
            grid=inputs.grid,
            boundary=boundary,
            n_steps=inputs.n_steps,
            dt=inputs.grid.dt,
            dx=inputs.grid.dx,
            eps_r=inputs.materials.eps_r,
            mu_r=inputs.materials.mu_r,
            zero_sigma=jnp.zeros_like(inputs.materials.sigma),
            debye_spec=debye_spec,
            cpml_axes=cpml_axes,
            pec_axes=inputs.pec_axes,
            pec_faces=pec_faces,
            cpml_params=cpml_params,
            src_waveforms_raw=src_waveforms_raw,
            src_meta=src_meta,
            prb_meta=prb_meta,
            initial_state=initial_state,
            inventory=inventory,
        )

    @classmethod
    def from_prepared_runner_state(
        cls,
        *,
        boundary: str,
        grid: Grid,
        prepared: Phase1HybridPreparedRunnerState,
        n_steps: int,
    ) -> Phase1HybridContext:
        return cls.from_inputs(
            Phase1HybridInputs.from_prepared_runner_state(
                boundary=boundary,
                grid=grid,
                prepared=prepared,
                n_steps=n_steps,
            )
        )

    @classmethod
    def from_inspected_runner_state(
        cls,
        *,
        boundary: str,
        probe_count: int,
        grid: Grid | None,
        prepared: Phase1HybridPreparedRunnerState | None,
        report: Phase1HybridInspection,
        n_steps: int | None,
    ) -> Phase1HybridContext:
        return Phase1HybridInputs.from_inspected_runner_state(
            boundary=boundary,
            probe_count=probe_count,
            grid=grid,
            prepared=prepared,
            report=report,
            n_steps=n_steps,
        ).require_context()

    def resolved_eps_r(self, eps_override: jnp.ndarray | None = None) -> jnp.ndarray:
        eps_r = self.eps_r if eps_override is None else eps_override
        if eps_r.shape != self.grid.shape:
            raise ValueError(
                f"eps_override shape {eps_r.shape} does not match context grid shape {self.grid.shape}"
            )
        return eps_r

    def run_time_series(self, eps_override: jnp.ndarray | None = None) -> jnp.ndarray:
        return run_phase1_forward_time_series(self, self.resolved_eps_r(eps_override))

    def forward_result(self, eps_override: jnp.ndarray | None = None) -> "ForwardResult":
        return phase1_forward_result(self.grid, self.run_time_series(eps_override))


@dataclass(frozen=True)
class Phase1HybridInputs:
    """Seam-owned input spec for Phase 1 hybrid preparation."""

    boundary: str
    periodic: tuple[bool, bool, bool]
    materials: MaterialArrays | None
    raw_sources: list[tuple[int, int, int, str, jnp.ndarray]] | tuple[tuple[int, int, int, str, jnp.ndarray], ...]
    probes: list[ProbeSpec] | tuple[ProbeSpec, ...]
    debye_spec: tuple | None
    debye: tuple | None
    lorentz: tuple | None
    ntff_box: object | None
    waveguide_ports: list | tuple | None
    pec_mask: jnp.ndarray | None
    pec_occupancy: jnp.ndarray | None
    grid: Grid | None = None
    n_steps: int | None = None
    pec_axes: str = ""
    cpml_axes: str = ""
    pec_faces: tuple[str, ...] = ()
    n_warmup: int = 0
    checkpoint_every: int | None = None
    scan_source_count: int | None = None
    report_override: Phase1HybridInspection | None = field(default=None, repr=False)

    @classmethod
    def unsupported(
        cls,
        report: Phase1HybridInspection,
    ) -> Phase1HybridInputs:
        return cls(
            boundary=report.boundary,
            periodic=report.periodic,
            materials=None,
            raw_sources=(),
            probes=(),
            debye_spec=None,
            debye=None,
            lorentz=None,
            ntff_box=None,
            waveguide_ports=None,
            pec_mask=None,
            pec_occupancy=None,
            report_override=report,
        )

    @classmethod
    def nonuniform_unsupported(
        cls,
        *,
        probe_count: int,
        boundary: str,
    ) -> Phase1HybridInputs:
        return cls.unsupported(
            Phase1HybridInspection.nonuniform_unsupported(
                probe_count=probe_count,
                boundary=boundary,
            )
        )

    @classmethod
    def from_prepared_runner_state(
        cls,
        *,
        boundary: str,
        grid: Grid,
        prepared: Phase1HybridPreparedRunnerState,
        n_steps: int,
    ) -> Phase1HybridInputs:
        return cls(
            boundary=boundary,
            periodic=prepared.periodic_bool,
            materials=prepared.materials,
            raw_sources=prepared.raw_phase1_sources,
            probes=prepared.probes,
            debye_spec=prepared.debye_spec,
            debye=prepared.debye,
            lorentz=prepared.lorentz,
            ntff_box=prepared.ntff_box,
            waveguide_ports=prepared.waveguide_ports,
            pec_mask=prepared.pec_mask_local,
            pec_occupancy=prepared.pec_occupancy_local,
            grid=grid,
            n_steps=n_steps,
            pec_axes=prepared.pec_axes_run,
            cpml_axes=prepared.cpml_axes_run,
            pec_faces=tuple(sorted(getattr(grid, "pec_faces", set()))),
            scan_source_count=len(prepared.sources),
        )

    @classmethod
    def from_inspected_runner_state(
        cls,
        *,
        boundary: str,
        probe_count: int,
        grid: Grid | None,
        prepared: Phase1HybridPreparedRunnerState | None,
        report: Phase1HybridInspection,
        n_steps: int | None,
    ) -> Phase1HybridInputs:
        if prepared is None:
            return cls.unsupported(report)
        assert grid is not None
        assert n_steps is not None
        return cls.from_prepared_runner_state(
            boundary=boundary,
            grid=grid,
            prepared=prepared,
            n_steps=n_steps,
        )

    @property
    def source_count(self) -> int:
        return self.report_override.source_count if self.report_override is not None else len(self.raw_sources)

    @property
    def probe_count(self) -> int:
        return self.report_override.probe_count if self.report_override is not None else len(self.probes)

    @cached_property
    def report(self) -> Phase1HybridInspection:
        return self.report_override or _phase1_hybrid_report_from_inputs(self)

    @property
    def supported(self) -> bool:
        return self.report.supported

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.report.reasons

    @property
    def reason_text(self) -> str:
        return self.report.reason_text

    @property
    def inventory(self) -> Phase1HybridInventory | None:
        return self.report.inventory

    def inspect(self) -> Phase1HybridInspection:
        return self.report

    def require_supported(self) -> Phase1HybridInputs:
        self.report.raise_for_unsupported()
        return self

    @cached_property
    def prepared_bundle(self) -> Phase1HybridPrepared:
        return prepare_phase1_hybrid(self)

    @cached_property
    def context(self) -> Phase1HybridContext | None:
        return self.prepared_bundle.context

    def require_context(self) -> Phase1HybridContext:
        self.require_supported()
        assert self.context is not None
        return self.context

    def prepare(self) -> Phase1HybridPrepared:
        return self.prepared_bundle

    def run_time_series(self, eps_override: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.require_context().run_time_series(eps_override)

    def forward_result(self, eps_override: jnp.ndarray | None = None) -> "ForwardResult":
        return self.require_context().forward_result(eps_override)


@dataclass(frozen=True)
class Phase1HybridInspection:
    """Stable inspection surface for the current Phase 1 hybrid seam."""

    supported: bool
    reasons: tuple[str, ...]
    inventory: Phase1HybridInventory | None
    source_count: int
    probe_count: int
    boundary: str
    periodic: tuple[bool, bool, bool]

    @classmethod
    def unsupported(
        cls,
        *,
        reasons: tuple[str, ...],
        source_count: int,
        probe_count: int,
        boundary: str,
        periodic: tuple[bool, bool, bool],
    ) -> Phase1HybridInspection:
        return cls(
            supported=False,
            reasons=reasons,
            inventory=None,
            source_count=source_count,
            probe_count=probe_count,
            boundary=boundary,
            periodic=periodic,
        )

    @classmethod
    def from_inputs(cls, inputs: Phase1HybridInputs) -> Phase1HybridInspection:
        if inputs.report_override is not None:
            return inputs.report_override
        report = inspect_phase1_hybrid(
                boundary=inputs.boundary,
                periodic=inputs.periodic,
                materials=inputs.materials,
                raw_sources=inputs.raw_sources,
                probes=inputs.probes,
                debye_spec=inputs.debye_spec,
                debye=inputs.debye,
                lorentz=inputs.lorentz,
            ntff_box=inputs.ntff_box,
            waveguide_ports=inputs.waveguide_ports,
            pec_mask=inputs.pec_mask,
            pec_occupancy=inputs.pec_occupancy,
            grid=inputs.grid,
            n_steps=inputs.n_steps,
            pec_axes=inputs.pec_axes,
            cpml_axes=inputs.cpml_axes,
            pec_faces=inputs.pec_faces,
            n_warmup=inputs.n_warmup,
            checkpoint_every=inputs.checkpoint_every,
        )
        if inputs.scan_source_count is not None and len(inputs.raw_sources) != inputs.scan_source_count:
            reasons = report.reasons + (
                "only add_source()-style J-sources are supported in Phase 1 hybrid mode",
            )
            report = cls.unsupported(
                reasons=reasons,
                source_count=report.source_count,
                probe_count=report.probe_count,
                boundary=report.boundary,
                periodic=report.periodic,
            )
        return report

    @classmethod
    def nonuniform_unsupported(
        cls,
        *,
        probe_count: int,
        boundary: str,
    ) -> Phase1HybridInspection:
        return cls.unsupported(
            reasons=("non-uniform grids are unsupported",),
            source_count=0,
            probe_count=probe_count,
            boundary=boundary,
            periodic=(False, False, False),
        )

    @classmethod
    def from_prepared_runner_state(
        cls,
        *,
        boundary: str,
        grid: Grid,
        prepared: Phase1HybridPreparedRunnerState,
        n_steps: int,
    ) -> Phase1HybridInspection:
        return Phase1HybridInputs.from_prepared_runner_state(
            boundary=boundary,
            grid=grid,
            prepared=prepared,
            n_steps=n_steps,
        ).inspect()

    @classmethod
    def from_inspected_runner_state(
        cls,
        *,
        boundary: str,
        probe_count: int,
        grid: Grid | None,
        prepared: Phase1HybridPreparedRunnerState | None,
        report: Phase1HybridInspection,
        n_steps: int | None,
    ) -> Phase1HybridInspection:
        if prepared is None:
            return cls.nonuniform_unsupported(
                probe_count=probe_count,
                boundary=boundary,
            )
        assert grid is not None
        assert n_steps is not None
        return cls.from_prepared_runner_state(
            boundary=boundary,
            grid=grid,
            prepared=prepared,
            n_steps=n_steps,
        )

    @property
    def reason_text(self) -> str:
        return "" if self.supported else "; ".join(self.reasons)

    def raise_for_unsupported(self) -> None:
        if not self.supported:
            raise ValueError(self.reason_text)


@dataclass(frozen=True)
class Phase1HybridPrepared:
    """Public prepared bundle for the supported Phase 1 seam."""

    report: Phase1HybridInspection
    context: Phase1HybridContext | None

    @classmethod
    def unsupported(cls, report: Phase1HybridInspection) -> Phase1HybridPrepared:
        return cls(report=report, context=None)

    @classmethod
    def nonuniform_unsupported(
        cls,
        *,
        probe_count: int,
        boundary: str,
    ) -> Phase1HybridPrepared:
        return cls.unsupported(
            Phase1HybridInspection.nonuniform_unsupported(
                probe_count=probe_count,
                boundary=boundary,
            )
        )

    @classmethod
    def from_inputs(cls, inputs: Phase1HybridInputs) -> Phase1HybridPrepared:
        report = inputs.inspect()
        context = None
        if report.supported:
            context = Phase1HybridContext.from_inputs(inputs)
        return cls(report=report, context=context)

    @classmethod
    def from_prepared_runner_state(
        cls,
        *,
        boundary: str,
        grid: Grid,
        prepared: Phase1HybridPreparedRunnerState,
        n_steps: int,
    ) -> Phase1HybridPrepared:
        return cls.from_inputs(
            Phase1HybridInputs.from_prepared_runner_state(
                boundary=boundary,
                grid=grid,
                prepared=prepared,
                n_steps=n_steps,
            )
        )

    @classmethod
    def from_inspected_runner_state(
        cls,
        *,
        boundary: str,
        probe_count: int,
        grid: Grid | None,
        prepared: Phase1HybridPreparedRunnerState | None,
        report: Phase1HybridInspection,
        n_steps: int | None,
    ) -> Phase1HybridPrepared:
        if prepared is None:
            return cls.unsupported(report)
        assert grid is not None
        assert n_steps is not None
        return cls.from_prepared_runner_state(
            boundary=boundary,
            grid=grid,
            prepared=prepared,
            n_steps=n_steps,
        )

    @property
    def supported(self) -> bool:
        return self.report.supported

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.report.reasons

    @property
    def reason_text(self) -> str:
        return self.report.reason_text

    @property
    def inventory(self) -> Phase1HybridInventory | None:
        return self.report.inventory

    @property
    def source_count(self) -> int:
        return self.report.source_count

    @property
    def probe_count(self) -> int:
        return self.report.probe_count

    @property
    def boundary(self) -> str:
        return self.report.boundary

    @property
    def periodic(self) -> tuple[bool, bool, bool]:
        return self.report.periodic

    @property
    def grid(self) -> Grid | None:
        return None if self.context is None else self.context.grid

    @property
    def eps_r(self) -> jnp.ndarray | None:
        return None if self.context is None else self.context.eps_r

    def require_supported(self) -> Phase1HybridPrepared:
        self.report.raise_for_unsupported()
        return self

    def require_context(self) -> Phase1HybridContext:
        self.require_supported()
        assert self.context is not None
        return self.context

    def run_time_series(self, eps_override: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.require_context().run_time_series(eps_override)

    def forward_result(self, eps_override: jnp.ndarray | None = None) -> "ForwardResult":
        return self.require_context().forward_result(eps_override)


def unsupported_phase1_hybrid_nonuniform(*, probe_count: int, boundary: str) -> Phase1HybridPrepared:
    return Phase1HybridPrepared.nonuniform_unsupported(
        probe_count=probe_count,
        boundary=boundary,
    )


def unsupported_phase1_hybrid_nonuniform_report(*, probe_count: int, boundary: str) -> Phase1HybridInspection:
    return Phase1HybridInspection.nonuniform_unsupported(
        probe_count=probe_count,
        boundary=boundary,
    )


def unsupported_phase1_hybrid_nonuniform_inputs(*, probe_count: int, boundary: str) -> Phase1HybridInputs:
    return Phase1HybridInputs.nonuniform_unsupported(
        probe_count=probe_count,
        boundary=boundary,
    )


def phase1_hybrid_support_reasons(
    *,
    boundary: str,
    periodic: tuple[bool, bool, bool],
    materials: MaterialArrays,
    sources: list[SourceSpec] | tuple[SourceSpec, ...] | None,
    probes: list[ProbeSpec] | tuple[ProbeSpec, ...] | None,
    debye_spec: tuple | None,
    debye: tuple | None,
    lorentz: tuple | None,
    ntff_box: object | None,
    waveguide_ports: list | tuple | None,
    pec_mask: jnp.ndarray | None,
    pec_occupancy: jnp.ndarray | None,
    n_warmup: int = 0,
    checkpoint_every: int | None = None,
) -> tuple[str, ...]:
    """Return explicit reasons that the Phase 1 hybrid lane is unsupported."""

    reasons: list[str] = []
    if boundary not in {"pec", "cpml"}:
        reasons.append(f"boundary={boundary!r} is unsupported")
    if periodic != (False, False, False):
        reasons.append("periodic axes are unsupported")
    if debye is not None and debye_spec is None:
        reasons.append("Debye reconstruction metadata is unavailable")
    if lorentz is not None:
        reasons.append("Lorentz dispersion is unsupported")
    if ntff_box is not None:
        reasons.append("NTFF accumulation is unsupported")
    if waveguide_ports:
        reasons.append("waveguide/wire/floquet port accumulation is unsupported")
    if pec_mask is not None:
        reasons.append("pec_mask replay is unsupported")
    if pec_occupancy is not None:
        reasons.append("pec_occupancy replay is unsupported")
    if n_warmup:
        reasons.append("n_warmup is unsupported")
    if checkpoint_every is not None:
        reasons.append("checkpoint_every is unsupported")
    if not sources:
        reasons.append("at least one source is required")
    if not probes:
        reasons.append("at least one probe is required")
    sigma = np.asarray(materials.sigma)
    if np.any(np.abs(sigma) > 0.0):
        reasons.append("lossy materials / port-loaded conductivity are unsupported")
    return tuple(reasons)


def prepare_phase1_hybrid_from_inspected_runner_state(
    *,
    boundary: str,
    probe_count: int,
    grid: Grid | None,
    prepared: Phase1HybridPreparedRunnerState | None,
    report: Phase1HybridInspection,
    n_steps: int | None,
) -> Phase1HybridPrepared:
    """Translate inspected runner state into the canonical Phase 1 bundle."""

    return Phase1HybridPrepared.from_inspected_runner_state(
        boundary=boundary,
        probe_count=probe_count,
        grid=grid,
        prepared=prepared,
        report=report,
        n_steps=n_steps,
    )


def prepare_phase1_hybrid_from_prepared_runner_state(
    *,
    boundary: str,
    grid: Grid,
    prepared: Phase1HybridPreparedRunnerState,
    n_steps: int,
) -> Phase1HybridPrepared:
    """Translate runner-prepared state into the canonical Phase 1 bundle."""

    return Phase1HybridPrepared.from_prepared_runner_state(
        boundary=boundary,
        grid=grid,
        prepared=prepared,
        n_steps=n_steps,
    )


def inspect_phase1_hybrid_from_inspected_runner_state(
    *,
    boundary: str,
    probe_count: int,
    grid: Grid | None,
    prepared: Phase1HybridPreparedRunnerState | None,
    report: Phase1HybridInspection,
    n_steps: int | None,
) -> Phase1HybridInspection:
    """Translate inspected runner state into the canonical Phase 1 report."""

    return Phase1HybridInspection.from_inspected_runner_state(
        boundary=boundary,
        probe_count=probe_count,
        grid=grid,
        prepared=prepared,
        report=report,
        n_steps=n_steps,
    )


def inspect_phase1_hybrid_from_prepared_runner_state(
    *,
    boundary: str,
    grid: Grid,
    prepared: Phase1HybridPreparedRunnerState,
    n_steps: int,
) -> Phase1HybridInspection:
    """Translate runner-prepared state into the canonical Phase 1 report."""

    return Phase1HybridInspection.from_prepared_runner_state(
        boundary=boundary,
        grid=grid,
        prepared=prepared,
        n_steps=n_steps,
    )


def _phase1_hybrid_report_from_inputs(inputs: Phase1HybridInputs) -> Phase1HybridInspection:
    return Phase1HybridInspection.from_inputs(inputs)


def inspect_phase1_hybrid_from_inputs(inputs: Phase1HybridInputs) -> Phase1HybridInspection:
    """Inspect the Phase 1 seam from the seam-owned input surface."""

    return inputs.inspect()


def prepare_phase1_hybrid_from_inputs(inputs: Phase1HybridInputs) -> Phase1HybridPrepared:
    """Prepare the Phase 1 seam from the seam-owned input surface."""

    return inputs.prepare()


def build_phase1_hybrid_context_from_inputs(inputs: Phase1HybridInputs) -> Phase1HybridContext:
    """Build the Phase 1 replay context from the seam-owned input surface."""

    return inputs.require_context()


def forward_phase1_hybrid_from_context(
    context: Phase1HybridContext,
    eps_override: jnp.ndarray | None = None,
) -> "ForwardResult":
    """Execute the Phase 1 seam from the seam-owned context surface."""

    return context.forward_result(eps_override)


def forward_phase1_hybrid_from_prepared(
    prepared: Phase1HybridPrepared,
    eps_override: jnp.ndarray | None = None,
) -> "ForwardResult":
    """Execute the Phase 1 seam from the seam-owned prepared bundle surface."""

    return prepared.forward_result(eps_override)


def forward_phase1_hybrid_from_inputs(
    inputs: Phase1HybridInputs,
    eps_override: jnp.ndarray | None = None,
) -> "ForwardResult":
    """Execute the Phase 1 seam from the seam-owned input surface."""

    return inputs.forward_result(eps_override)


def forward_phase1_hybrid_from_prepared_runner_state(
    *,
    boundary: str,
    grid: Grid,
    prepared: Phase1HybridPreparedRunnerState,
    n_steps: int,
    eps_override: jnp.ndarray | None = None,
) -> "ForwardResult":
    """Execute the Phase 1 seam from the seam-owned prepared-runner surface."""

    return forward_phase1_hybrid_from_context(
        build_phase1_hybrid_context_from_prepared_runner_state(
            boundary=boundary,
            grid=grid,
            prepared=prepared,
            n_steps=n_steps,
        ),
        eps_override,
    )


def forward_phase1_hybrid_from_inspected_runner_state(
    *,
    boundary: str,
    probe_count: int,
    grid: Grid | None,
    prepared: Phase1HybridPreparedRunnerState | None,
    report: Phase1HybridInspection,
    n_steps: int | None,
    eps_override: jnp.ndarray | None = None,
) -> "ForwardResult":
    """Execute the Phase 1 seam from the seam-owned inspected-runner surface."""

    return forward_phase1_hybrid_from_context(
        build_phase1_hybrid_context_from_inspected_runner_state(
            boundary=boundary,
            probe_count=probe_count,
            grid=grid,
            prepared=prepared,
            report=report,
            n_steps=n_steps,
        ),
        eps_override,
    )


def prepare_phase1_hybrid(inputs: Phase1HybridInputs) -> Phase1HybridPrepared:
    """Build the canonical public prepared bundle for Phase 1 hybrid runs."""

    return Phase1HybridPrepared.from_inputs(inputs)


def inspect_phase1_hybrid(
    *,
    boundary: str,
    periodic: tuple[bool, bool, bool],
    materials: MaterialArrays,
    raw_sources: list[tuple[int, int, int, str, jnp.ndarray]] | tuple[tuple[int, int, int, str, jnp.ndarray], ...],
    probes: list[ProbeSpec] | tuple[ProbeSpec, ...],
    debye_spec: tuple | None,
    debye: tuple | None,
    lorentz: tuple | None,
    ntff_box: object | None,
    waveguide_ports: list | tuple | None,
    pec_mask: jnp.ndarray | None,
    pec_occupancy: jnp.ndarray | None,
    grid: Grid | None = None,
    n_steps: int | None = None,
    pec_axes: str = "",
    cpml_axes: str = "",
    pec_faces: tuple[str, ...] = (),
    n_warmup: int = 0,
    checkpoint_every: int | None = None,
) -> Phase1HybridInspection:
    """Inspect whether the current configuration fits the Phase 1 seam."""

    reasons = phase1_hybrid_support_reasons(
        boundary=boundary,
        periodic=periodic,
        materials=materials,
        sources=raw_sources,
        probes=probes,
        debye_spec=debye_spec,
        debye=debye,
        lorentz=lorentz,
        ntff_box=ntff_box,
        waveguide_ports=waveguide_ports,
        pec_mask=pec_mask,
        pec_occupancy=pec_occupancy,
        n_warmup=n_warmup,
        checkpoint_every=checkpoint_every,
    )
    inventory = None
    if not reasons and grid is not None and n_steps is not None:
        inventory = Phase1HybridContext.from_inputs(
            Phase1HybridInputs(
                boundary=boundary,
                periodic=periodic,
                materials=materials,
                raw_sources=raw_sources,
                probes=probes,
                debye_spec=debye_spec,
                debye=debye,
                lorentz=lorentz,
                ntff_box=ntff_box,
                waveguide_ports=waveguide_ports,
                pec_mask=pec_mask,
                pec_occupancy=pec_occupancy,
                grid=grid,
                n_steps=n_steps,
                pec_axes=pec_axes,
                cpml_axes=cpml_axes,
                pec_faces=pec_faces,
                n_warmup=n_warmup,
                checkpoint_every=checkpoint_every,
            )
        ).inventory
    return Phase1HybridInspection(
        supported=not reasons,
        reasons=reasons,
        inventory=inventory,
        source_count=len(raw_sources),
        probe_count=len(probes),
        boundary=boundary,
        periodic=periodic,
    )


def build_phase1_hybrid_inputs_from_prepared_runner_state(
    *,
    boundary: str,
    grid: Grid,
    prepared: Phase1HybridPreparedRunnerState,
    n_steps: int,
) -> Phase1HybridInputs:
    """Translate runner-prepared state into the canonical Phase 1 input spec."""

    return Phase1HybridInputs.from_prepared_runner_state(
        boundary=boundary,
        grid=grid,
        prepared=prepared,
        n_steps=n_steps,
    )


def build_phase1_hybrid_inputs_from_inspected_runner_state(
    *,
    boundary: str,
    probe_count: int,
    grid: Grid | None,
    prepared: Phase1HybridPreparedRunnerState | None,
    report: Phase1HybridInspection,
    n_steps: int | None,
) -> Phase1HybridInputs:
    """Translate inspected runner state into the canonical Phase 1 input spec."""

    return Phase1HybridInputs.from_inspected_runner_state(
        boundary=boundary,
        probe_count=probe_count,
        grid=grid,
        prepared=prepared,
        report=report,
        n_steps=n_steps,
    )


def build_phase1_hybrid_context_from_prepared_runner_state(
    *,
    boundary: str,
    grid: Grid,
    prepared: Phase1HybridPreparedRunnerState,
    n_steps: int,
) -> Phase1HybridContext:
    """Translate runner-prepared state into the canonical Phase 1 context."""

    return Phase1HybridContext.from_prepared_runner_state(
        boundary=boundary,
        grid=grid,
        prepared=prepared,
        n_steps=n_steps,
    )


def build_phase1_hybrid_context_from_inspected_runner_state(
    *,
    boundary: str,
    probe_count: int,
    grid: Grid | None,
    prepared: Phase1HybridPreparedRunnerState | None,
    report: Phase1HybridInspection,
    n_steps: int | None,
) -> Phase1HybridContext:
    """Translate inspected runner state into the canonical Phase 1 context."""

    return Phase1HybridContext.from_inspected_runner_state(
        boundary=boundary,
        probe_count=probe_count,
        grid=grid,
        prepared=prepared,
        report=report,
        n_steps=n_steps,
    )


def build_phase1_hybrid_context(
    *,
    grid: Grid,
    materials: MaterialArrays,
    n_steps: int,
    raw_sources: list[tuple[int, int, int, str, jnp.ndarray]] | tuple[tuple[int, int, int, str, jnp.ndarray], ...],
    probes: list[ProbeSpec] | tuple[ProbeSpec, ...],
    pec_axes: str,
    debye_spec: tuple | None = None,
    boundary: str = "pec",
    cpml_axes: str | None = None,
    pec_faces: tuple[str, ...] | None = None,
) -> Phase1HybridContext:
    """Build the static replay context for the Phase 1 hybrid seam."""

    return Phase1HybridContext.from_inputs(
        Phase1HybridInputs(
            boundary=boundary,
            periodic=(False, False, False),
            materials=materials,
            raw_sources=raw_sources,
            probes=probes,
            debye_spec=debye_spec,
            debye=None,
            lorentz=None,
            ntff_box=None,
            waveguide_ports=None,
            pec_mask=None,
            pec_occupancy=None,
            grid=grid,
            n_steps=n_steps,
            pec_axes=pec_axes,
            cpml_axes=grid.cpml_axes if cpml_axes is None else cpml_axes,
            pec_faces=tuple(sorted(getattr(grid, "pec_faces", set()))) if pec_faces is None else pec_faces,
        )
    )


def run_phase1_forward_time_series(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
) -> jnp.ndarray:
    """Run the narrow uniform/lossless forward seam."""

    materials = _materials_from_eps(context, eps_r)
    src_waveforms = _source_waveforms_from_eps(context, eps_r)
    if context.debye_spec is not None:
        debye_coeffs, debye_state = _runtime_debye(context.dt, materials, context.debye_spec)
        if context.boundary == "cpml":
            _, _, _, time_series = _run_uniform_debye_cpml_time_series(
                context,
                materials,
                debye_coeffs,
                debye_state,
                src_waveforms,
            )
            return time_series
        if context.boundary == "pec":
            _, _, time_series = _run_uniform_debye_pec_time_series(
                context,
                materials,
                debye_coeffs,
                debye_state,
                src_waveforms,
            )
            return time_series
        raise ValueError(f"boundary={context.boundary!r} is unsupported")
    if context.boundary == "cpml":
        _, _, time_series = _run_uniform_lossless_cpml_time_series(context, materials, src_waveforms)
        return time_series
    if context.boundary != "pec":
        raise ValueError(f"boundary={context.boundary!r} is unsupported")
    coeffs = _coeffs_from_materials(context, materials)
    _, time_series = _run_uniform_lossless_pec_time_series(context, coeffs, src_waveforms)
    return time_series


def make_phase1_hybrid_forward(context: Phase1HybridContext) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a custom_vjp forward op for the Phase 1 hybrid seam."""

    @jax.custom_vjp
    def _forward_time_series(eps_r: jnp.ndarray) -> jnp.ndarray:
        return run_phase1_forward_time_series(context, eps_r)

    def _forward_fwd(eps_r: jnp.ndarray):
        materials = _materials_from_eps(context, eps_r)
        src_waveforms = _source_waveforms_from_eps(context, eps_r)
        if context.debye_spec is not None:
            debye_coeffs, debye_state = _runtime_debye(context.dt, materials, context.debye_spec)
            if context.boundary == "cpml":
                _, _, _, time_series, states_before, cpml_states_before, debye_states_before = (
                    _run_uniform_debye_cpml_time_series(
                        context,
                        materials,
                        debye_coeffs,
                        debye_state,
                        src_waveforms,
                        return_trace=True,
                    )
                )
                return time_series, (
                    eps_r,
                    states_before,
                    cpml_states_before,
                    debye_states_before,
                    context.src_waveforms_raw,
                )
            if context.boundary == "pec":
                _, _, time_series, states_before, debye_states_before = _run_uniform_debye_pec_time_series(
                    context,
                    materials,
                    debye_coeffs,
                    debye_state,
                    src_waveforms,
                    return_trace=True,
                )
                return time_series, (eps_r, states_before, debye_states_before, context.src_waveforms_raw)
            raise ValueError(f"boundary={context.boundary!r} is unsupported")
        if context.boundary == "cpml":
            _, _, time_series, states_before, cpml_states_before = _run_uniform_lossless_cpml_time_series(
                context,
                materials,
                src_waveforms,
                return_trace=True,
            )
            return time_series, (eps_r, states_before, cpml_states_before, context.src_waveforms_raw)
        _, time_series, states_before = _run_uniform_lossless_pec_time_series(
            context,
            _coeffs_from_materials(context, materials),
            src_waveforms,
            return_trace=True,
        )
        return time_series, (eps_r, states_before, context.src_waveforms_raw)

    def _forward_bwd(res, probe_bar: jnp.ndarray):
        if context.debye_spec is not None and context.boundary == "cpml":
            eps_r, states_before, cpml_states_before, debye_states_before, raw_src_waveforms = res
            zero_state = _zeros_like_state(context.initial_state)
            zero_cpml = _zero_cpml_state(context.grid)
            zero_debye = _zero_debye_state(
                _runtime_debye(context.dt, _materials_from_eps(context, eps_r), context.debye_spec)[1]
            )
            zero_eps = jnp.zeros_like(eps_r)

            def reverse_step(carry, xs):
                lambda_next, lambda_cpml_next, lambda_debye_next, grad_eps = carry
                state_before, cpml_before, debye_before, raw_src_vals, probe_cot = xs

                def step_from_eps(state, cpml_state, debye_state, eps_local):
                    materials_local = _materials_from_eps(context, eps_local)
                    debye_coeffs, _ = _runtime_debye(
                        context.dt,
                        materials_local,
                        context.debye_spec,
                    )
                    return _uniform_debye_cpml_step(
                        state,
                        cpml_state,
                        debye_state,
                        _source_values_from_eps(context, eps_local, raw_src_vals),
                        materials_local,
                        debye_coeffs,
                        context.grid,
                        context.cpml_params,
                        context.cpml_axes,
                        context.pec_axes,
                        context.pec_faces,
                        context.src_meta,
                        context.prb_meta,
                    )

                _, step_vjp = jax.vjp(step_from_eps, state_before, cpml_before, debye_before, eps_r)
                lambda_before, lambda_cpml_before, lambda_debye_before, grad_eps_step = step_vjp(
                    (lambda_next, lambda_cpml_next, lambda_debye_next, probe_cot)
                )
                return (
                    lambda_before,
                    lambda_cpml_before,
                    lambda_debye_before,
                    grad_eps + grad_eps_step,
                ), None

            (_, _, _, grad_eps), _ = jax.lax.scan(
                reverse_step,
                (zero_state, zero_cpml, zero_debye, zero_eps),
                (states_before, cpml_states_before, debye_states_before, raw_src_waveforms, probe_bar),
                reverse=True,
            )
            return (grad_eps,)

        if context.debye_spec is not None and context.boundary == "pec":
            eps_r, states_before, debye_states_before, raw_src_waveforms = res
            zero_state = _zeros_like_state(context.initial_state)
            zero_debye = _zero_debye_state(
                _runtime_debye(context.dt, _materials_from_eps(context, eps_r), context.debye_spec)[1]
            )
            zero_eps = jnp.zeros_like(eps_r)

            def reverse_step(carry, xs):
                lambda_next, lambda_debye_next, grad_eps = carry
                state_before, debye_before, raw_src_vals, probe_cot = xs

                def step_from_eps(state, debye_state, eps_local):
                    materials_local = _materials_from_eps(context, eps_local)
                    debye_coeffs, _ = _runtime_debye(
                        context.dt,
                        materials_local,
                        context.debye_spec,
                    )
                    return _uniform_debye_pec_step(
                        state,
                        debye_state,
                        _source_values_from_eps(context, eps_local, raw_src_vals),
                        materials_local,
                        debye_coeffs,
                        context.grid,
                        context.pec_axes,
                        context.pec_faces,
                        context.src_meta,
                        context.prb_meta,
                    )

                _, step_vjp = jax.vjp(step_from_eps, state_before, debye_before, eps_r)
                lambda_before, lambda_debye_before, grad_eps_step = step_vjp(
                    (lambda_next, lambda_debye_next, probe_cot)
                )
                return (lambda_before, lambda_debye_before, grad_eps + grad_eps_step), None

            (_, _, grad_eps), _ = jax.lax.scan(
                reverse_step,
                (zero_state, zero_debye, zero_eps),
                (states_before, debye_states_before, raw_src_waveforms, probe_bar),
                reverse=True,
            )
            return (grad_eps,)

        if context.boundary == "cpml":
            eps_r, states_before, cpml_states_before, raw_src_waveforms = res
            zero_state = _zeros_like_state(context.initial_state)
            zero_cpml = _zero_cpml_state(context.grid)
            zero_eps = jnp.zeros_like(eps_r)

            def reverse_step(carry, xs):
                lambda_next, lambda_cpml_next, grad_eps = carry
                state_before, cpml_before, raw_src_vals, probe_cot = xs

                def step_from_eps(state, cpml_state, eps_local):
                    return _uniform_lossless_cpml_step(
                        state,
                        cpml_state,
                        _source_values_from_eps(context, eps_local, raw_src_vals),
                        _materials_from_eps(context, eps_local),
                        context.grid,
                        context.cpml_params,
                        context.cpml_axes,
                        context.pec_axes,
                        context.pec_faces,
                        context.src_meta,
                        context.prb_meta,
                    )

                _, step_vjp = jax.vjp(step_from_eps, state_before, cpml_before, eps_r)
                lambda_before, lambda_cpml_before, grad_eps_step = step_vjp(
                    (lambda_next, lambda_cpml_next, probe_cot)
                )
                return (lambda_before, lambda_cpml_before, grad_eps + grad_eps_step), None

            (_, _, grad_eps), _ = jax.lax.scan(
                reverse_step,
                (zero_state, zero_cpml, zero_eps),
                (states_before, cpml_states_before, raw_src_waveforms, probe_bar),
                reverse=True,
            )
            return (grad_eps,)

        eps_r, states_before, raw_src_waveforms = res
        zero_state = _zeros_like_state(context.initial_state)
        zero_eps = jnp.zeros_like(eps_r)

        def reverse_step(carry, xs):
            lambda_next, grad_eps = carry
            state_before, raw_src_vals, probe_cot = xs

            def step_from_eps(state, eps_local):
                return _uniform_lossless_pec_step(
                    state,
                    _source_values_from_eps(context, eps_local, raw_src_vals),
                    _coeffs_from_eps(context, eps_local),
                    context.src_meta,
                    context.prb_meta,
                )

            _, step_vjp = jax.vjp(step_from_eps, state_before, eps_r)
            lambda_before, grad_eps_step = step_vjp((lambda_next, probe_cot))
            return (lambda_before, grad_eps + grad_eps_step), None

        (_, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (zero_state, zero_eps),
            (states_before, raw_src_waveforms, probe_bar),
            reverse=True,
        )
        return (grad_eps,)

    _forward_time_series.defvjp(_forward_fwd, _forward_bwd)
    return _forward_time_series


def _run_uniform_lossless_pec_time_series(
    context: Phase1HybridContext,
    coeffs: UpdateCoeffs,
    src_waveforms: jnp.ndarray,
    *,
    return_trace: bool = False,
):
    """Run the extracted time-series seam with precomputed coefficients."""

    def body(state, src_vals):
        next_state, probe_out = _uniform_lossless_pec_step(
            state,
            src_vals,
            coeffs,
            context.src_meta,
            context.prb_meta,
        )
        if return_trace:
            return next_state, (probe_out, state)
        return next_state, probe_out

    final_state, outputs = jax.lax.scan(body, context.initial_state, src_waveforms)
    if return_trace:
        time_series, states_before = outputs
        return final_state, time_series, states_before
    return final_state, outputs


def _run_uniform_lossless_cpml_time_series(
    context: Phase1HybridContext,
    materials: MaterialArrays,
    src_waveforms: jnp.ndarray,
    *,
    return_trace: bool = False,
):
    """Run the extracted time-series seam with CPML and PEC-face enforcement."""

    assert context.cpml_params is not None

    def body(carry, src_vals):
        state, cpml_state = carry
        next_state, next_cpml_state, probe_out = _uniform_lossless_cpml_step(
            state,
            cpml_state,
            src_vals,
            materials,
            context.grid,
            context.cpml_params,
            context.cpml_axes,
            context.pec_axes,
            context.pec_faces,
            context.src_meta,
            context.prb_meta,
        )
        if return_trace:
            return (next_state, next_cpml_state), (probe_out, state, cpml_state)
        return (next_state, next_cpml_state), probe_out

    (final_state, final_cpml_state), outputs = jax.lax.scan(
        body,
        (context.initial_state, _zero_cpml_state(context.grid)),
        src_waveforms,
    )
    if return_trace:
        time_series, states_before, cpml_states_before = outputs
        return final_state, final_cpml_state, time_series, states_before, cpml_states_before
    return final_state, final_cpml_state, outputs


def _run_uniform_debye_pec_time_series(
    context: Phase1HybridContext,
    materials: MaterialArrays,
    debye_coeffs: DebyeCoeffs,
    debye_state: DebyeState,
    src_waveforms: jnp.ndarray,
    *,
    return_trace: bool = False,
):
    """Run the extracted time-series seam with Debye and PEC enforcement."""

    def body(carry, src_vals):
        state, debye_state_local = carry
        next_state, next_debye_state, probe_out = _uniform_debye_pec_step(
            state,
            debye_state_local,
            src_vals,
            materials,
            debye_coeffs,
            context.grid,
            context.pec_axes,
            context.pec_faces,
            context.src_meta,
            context.prb_meta,
        )
        if return_trace:
            return (next_state, next_debye_state), (probe_out, state, debye_state_local)
        return (next_state, next_debye_state), probe_out

    (final_state, final_debye_state), outputs = jax.lax.scan(
        body,
        (context.initial_state, debye_state),
        src_waveforms,
    )
    if return_trace:
        time_series, states_before, debye_states_before = outputs
        return final_state, final_debye_state, time_series, states_before, debye_states_before
    return final_state, final_debye_state, outputs


def _run_uniform_debye_cpml_time_series(
    context: Phase1HybridContext,
    materials: MaterialArrays,
    debye_coeffs: DebyeCoeffs,
    debye_state: DebyeState,
    src_waveforms: jnp.ndarray,
    *,
    return_trace: bool = False,
):
    """Run the extracted time-series seam with Debye, CPML, and PEC enforcement."""

    assert context.cpml_params is not None

    def body(carry, src_vals):
        state, cpml_state, debye_state_local = carry
        next_state, next_cpml_state, next_debye_state, probe_out = _uniform_debye_cpml_step(
            state,
            cpml_state,
            debye_state_local,
            src_vals,
            materials,
            debye_coeffs,
            context.grid,
            context.cpml_params,
            context.cpml_axes,
            context.pec_axes,
            context.pec_faces,
            context.src_meta,
            context.prb_meta,
        )
        if return_trace:
            return (next_state, next_cpml_state, next_debye_state), (
                probe_out,
                state,
                cpml_state,
                debye_state_local,
            )
        return (next_state, next_cpml_state, next_debye_state), probe_out

    (final_state, final_cpml_state, final_debye_state), outputs = jax.lax.scan(
        body,
        (context.initial_state, _zero_cpml_state(context.grid), debye_state),
        src_waveforms,
    )
    if return_trace:
        time_series, states_before, cpml_states_before, debye_states_before = outputs
        return (
            final_state,
            final_cpml_state,
            final_debye_state,
            time_series,
            states_before,
            cpml_states_before,
            debye_states_before,
        )
    return final_state, final_cpml_state, final_debye_state, outputs


def _uniform_lossless_pec_step(
    state: Phase1FieldState,
    src_vals: jnp.ndarray,
    coeffs: UpdateCoeffs,
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> tuple[Phase1FieldState, jnp.ndarray]:
    """Extracted Phase 1 scan step: fast Yee step + source injection + probes."""

    fdtd = update_he_fast(_to_fdtd(state), coeffs)
    next_state = _from_fdtd(fdtd)
    next_state = _inject_sources(next_state, src_vals, src_meta)
    probe_out = _sample_probes(next_state, prb_meta)
    return next_state, probe_out


def _uniform_lossless_cpml_step(
    state: Phase1FieldState,
    cpml_state: CPMLState,
    src_vals: jnp.ndarray,
    materials: MaterialArrays,
    grid: Grid,
    cpml_params: CPMLAxisParams,
    cpml_axes: str,
    pec_axes: str,
    pec_faces: tuple[str, ...],
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> tuple[Phase1FieldState, CPMLState, jnp.ndarray]:
    """Extracted CPML scan step: Yee + CPML + PEC faces + sources + probes."""

    fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=(False, False, False))
    fdtd, next_cpml_state = apply_cpml_h(
        fdtd,
        cpml_params,
        cpml_state,
        grid,
        cpml_axes,
        materials=materials,
    )
    fdtd = update_e(fdtd, materials, grid.dt, grid.dx, periodic=(False, False, False))
    fdtd, next_cpml_state = apply_cpml_e(
        fdtd,
        cpml_params,
        next_cpml_state,
        grid,
        cpml_axes,
        materials=materials,
    )
    if pec_axes:
        fdtd = apply_pec(fdtd, axes=pec_axes)
    if pec_faces:
        fdtd = apply_pec_faces(fdtd, pec_faces)
    next_state = _from_fdtd(fdtd)
    next_state = _inject_sources(next_state, src_vals, src_meta)
    probe_out = _sample_probes(next_state, prb_meta)
    return next_state, next_cpml_state, probe_out


def _uniform_debye_pec_step(
    state: Phase1FieldState,
    debye_state: DebyeState,
    src_vals: jnp.ndarray,
    materials: MaterialArrays,
    debye_coeffs: DebyeCoeffs,
    grid: Grid,
    pec_axes: str,
    pec_faces: tuple[str, ...],
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> tuple[Phase1FieldState, DebyeState, jnp.ndarray]:
    """Extracted Debye scan step: Yee H + Debye E + PEC + sources + probes."""

    fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=(False, False, False))
    fdtd, next_debye_state = update_e_debye(
        fdtd,
        debye_coeffs,
        debye_state,
        grid.dt,
        grid.dx,
        periodic=(False, False, False),
    )
    if pec_axes:
        fdtd = apply_pec(fdtd, axes=pec_axes)
    if pec_faces:
        fdtd = apply_pec_faces(fdtd, pec_faces)
    next_state = _from_fdtd(fdtd)
    next_state = _inject_sources(next_state, src_vals, src_meta)
    probe_out = _sample_probes(next_state, prb_meta)
    return next_state, next_debye_state, probe_out


def _uniform_debye_cpml_step(
    state: Phase1FieldState,
    cpml_state: CPMLState,
    debye_state: DebyeState,
    src_vals: jnp.ndarray,
    materials: MaterialArrays,
    debye_coeffs: DebyeCoeffs,
    grid: Grid,
    cpml_params: CPMLAxisParams,
    cpml_axes: str,
    pec_axes: str,
    pec_faces: tuple[str, ...],
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> tuple[Phase1FieldState, CPMLState, DebyeState, jnp.ndarray]:
    """Extracted Debye+CPML scan step: Yee H + CPML + Debye E + PEC + sources + probes."""

    fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=(False, False, False))
    fdtd, next_cpml_state = apply_cpml_h(
        fdtd,
        cpml_params,
        cpml_state,
        grid,
        cpml_axes,
        materials=materials,
    )
    fdtd, next_debye_state = update_e_debye(
        fdtd,
        debye_coeffs,
        debye_state,
        grid.dt,
        grid.dx,
        periodic=(False, False, False),
    )
    fdtd, next_cpml_state = apply_cpml_e(
        fdtd,
        cpml_params,
        next_cpml_state,
        grid,
        cpml_axes,
        materials=materials,
    )
    if pec_axes:
        fdtd = apply_pec(fdtd, axes=pec_axes)
    if pec_faces:
        fdtd = apply_pec_faces(fdtd, pec_faces)
    next_state = _from_fdtd(fdtd)
    next_state = _inject_sources(next_state, src_vals, src_meta)
    probe_out = _sample_probes(next_state, prb_meta)
    return next_state, next_cpml_state, next_debye_state, probe_out


def _coeffs_from_eps(context: Phase1HybridContext, eps_r: jnp.ndarray) -> UpdateCoeffs:
    materials = _materials_from_eps(context, eps_r)
    return _coeffs_from_materials(context, materials)


def _materials_from_eps(context: Phase1HybridContext, eps_r: jnp.ndarray) -> MaterialArrays:
    return MaterialArrays(
        eps_r=eps_r,
        sigma=context.zero_sigma,
        mu_r=context.mu_r,
    )


def _coeffs_from_materials(context: Phase1HybridContext, materials: MaterialArrays) -> UpdateCoeffs:
    return precompute_coeffs(materials, context.dt, context.dx, pec_axes=context.pec_axes)


def _runtime_debye(
    dt: float,
    materials: MaterialArrays,
    debye_spec: tuple | None,
) -> tuple[DebyeCoeffs, DebyeState]:
    assert debye_spec is not None
    poles, masks = debye_spec
    return init_debye(poles, materials, dt, mask=masks)


def _source_waveforms_from_eps(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
) -> jnp.ndarray:
    if context.src_waveforms_raw.shape[1] == 0:
        return context.src_waveforms_raw
    scaled = []
    for idx, (i, j, k, _) in enumerate(context.src_meta):
        scaled.append(
            _source_values_from_eps(context, eps_r, context.src_waveforms_raw[:, idx], idx)
        )
    return jnp.stack(scaled, axis=-1)


def _source_values_from_eps(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
    raw_src_vals: jnp.ndarray,
    src_index: int | None = None,
) -> jnp.ndarray:
    if src_index is not None:
        i, j, k, _ = context.src_meta[src_index]
        eps = eps_r[i, j, k] * jnp.float32(EPS_0)
        cb = jnp.float32(context.dt) / eps
        return raw_src_vals * cb

    if raw_src_vals.shape[0] == 0:
        return raw_src_vals

    scaled = []
    for idx, (i, j, k, _) in enumerate(context.src_meta):
        eps = eps_r[i, j, k] * jnp.float32(EPS_0)
        cb = jnp.float32(context.dt) / eps
        scaled.append(raw_src_vals[idx] * cb)
    return jnp.stack(scaled)


def _inject_sources(
    state: Phase1FieldState,
    src_vals: jnp.ndarray,
    src_meta: tuple[tuple[int, int, int, str], ...],
) -> Phase1FieldState:
    if not src_meta:
        return state
    updated = state
    for idx_s, (si, sj, sk, component) in enumerate(src_meta):
        field = getattr(updated, component)
        field = field.at[si, sj, sk].add(src_vals[idx_s].astype(field.dtype))
        updated = updated._replace(**{component: field})
    return updated


def _sample_probes(
    state: Phase1FieldState,
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> jnp.ndarray:
    if not prb_meta:
        return jnp.zeros((0,), dtype=state.ex.dtype)
    samples = [getattr(state, component)[i, j, k] for i, j, k, component in prb_meta]
    return jnp.stack(samples)


def _to_fdtd(state: Phase1FieldState) -> FDTDState:
    return FDTDState(
        ex=state.ex,
        ey=state.ey,
        ez=state.ez,
        hx=state.hx,
        hy=state.hy,
        hz=state.hz,
        step=jnp.array(0, dtype=jnp.int32),
    )


def _from_fdtd(state: FDTDState) -> Phase1FieldState:
    return Phase1FieldState(
        ex=state.ex,
        ey=state.ey,
        ez=state.ez,
        hx=state.hx,
        hy=state.hy,
        hz=state.hz,
    )


def _zeros_like_state(state: Phase1FieldState) -> Phase1FieldState:
    return Phase1FieldState(*(jnp.zeros_like(field) for field in state))


def _zero_cpml_state(grid: Grid) -> CPMLState:
    _, cpml_state = init_cpml(grid, pec_faces=set(getattr(grid, "pec_faces", set())))
    return cpml_state


def _zero_debye_state(state: DebyeState) -> DebyeState:
    return DebyeState(
        px=jnp.zeros_like(state.px),
        py=jnp.zeros_like(state.py),
        pz=jnp.zeros_like(state.pz),
    )
