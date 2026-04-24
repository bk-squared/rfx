"""Hybrid adjoint helpers for the staged uniform dispersive time-series seam.

This module currently supports the narrow staged seam used by the hybrid
adjoint proof-of-concept and Phase 3A expansion work:

- uniform grid
- non-periodic
- lossless, Debye-dispersive, or Lorentz-dispersive materials with zero conductivity
- point-source / point-probe time-series objectives
- PEC boundaries
- CPML boundaries (including CPML + per-face PEC overrides)

Unsupported physics is expected to route back to the existing pure-AD
``Simulation.forward()`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, NamedTuple, TYPE_CHECKING

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
from rfx.farfield import NTFFData, init_ntff_data, accumulate_ntff
from rfx.grid import Grid
from rfx.materials.debye import DebyeCoeffs, DebyeState, init_debye, update_e_debye
from rfx.materials.lorentz import LorentzCoeffs, LorentzState, init_lorentz, update_e_lorentz
from rfx.nonuniform import NonUniformGrid, run_nonuniform
from rfx.simulation import ProbeSpec, SourceSpec

if TYPE_CHECKING:
    from rfx.api import ForwardResult


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
    port_metadata: object | None
    debye_spec: tuple | None
    debye: tuple | None
    lorentz_spec: tuple | None
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
            port_metadata=prepared.port_metadata,
            debye_spec=prepared.debye_spec,
            debye=prepared.debye,
            lorentz_spec=prepared.lorentz_spec,
            lorentz=prepared.lorentz,
            ntff_box=prepared.ntff_box,
            waveguide_ports=prepared.waveguide_ports,
            periodic_bool=prepared.periodic_bool,
            cpml_axes_run=prepared.cpml_axes_run,
            pec_mask_local=prepared.pec_mask_local,
            pec_occupancy_local=prepared.pec_occupancy_local,
            pec_axes_run=prepared.pec_axes_run,
        )


class _Phase1HybridObservables(NamedTuple):
    """Array-only observable bundle for hybrid custom_vjp branches."""

    time_series: jnp.ndarray
    ntff_data: NTFFData | None = None


def phase1_forward_result(
    grid: Grid,
    time_series: jnp.ndarray,
    ntff_data: object = None,
    ntff_box: object = None,
) -> "ForwardResult":
    """Build the minimal ForwardResult for the Phase 1 seam."""

    from rfx.api import ForwardResult

    return ForwardResult(
        time_series=time_series,
        ntff_data=ntff_data,
        ntff_box=ntff_box,
        grid=grid,
        s_params=None,
        freqs=None,
    )


@dataclass(frozen=True)
class Phase1HybridContext:
    """Static replay context for the experimental Phase 1 hybrid lane."""

    grid: Grid | NonUniformGrid
    boundary: str
    periodic: tuple[bool, bool, bool]
    n_steps: int
    dt: float
    dx: float
    eps_r: jnp.ndarray
    mu_r: jnp.ndarray
    sigma: jnp.ndarray
    debye_spec: tuple | None
    lorentz_spec: tuple | None
    cpml_axes: str
    pec_axes: str
    pec_faces: tuple[str, ...]
    cpml_params: CPMLAxisParams | None
    ntff_box: object | None
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
        if np.any(np.abs(np.asarray(inputs.materials.sigma)) > 0.0):
            replay_inputs.append("sigma")
        if inputs.port_metadata is not None and _supports_phase2_lumped_port_proxy_subset(inputs.port_metadata):
            replay_inputs.append("port_metadata")
        debye_spec = inputs.debye_spec
        lorentz_spec = inputs.lorentz_spec
        if debye_spec is not None and lorentz_spec is not None:
            raise ValueError("mixed Debye+Lorentz dispersion is unsupported")
        if inputs.lorentz is not None and lorentz_spec is None:
            raise ValueError("Lorentz reconstruction metadata is unavailable")
        if lorentz_spec is not None:
            lorentz_poles, _ = lorentz_spec
            if any(getattr(pole, "omega_0", None) == 0.0 for pole in lorentz_poles):
                raise ValueError("Drude-shaped Lorentz poles are unsupported")
        if debye_spec is not None:
            _, initial_debye_state = _runtime_debye(inputs.grid.dt, inputs.materials, debye_spec)
            carry_arrays.extend(
                (f"debye.{name}", arr)
                for name, arr in zip(initial_debye_state._fields, initial_debye_state)
            )
            replay_inputs.append("debye_spec")
            replay_outputs.append("final_debye_state")
        if lorentz_spec is not None:
            _, initial_lorentz_state = _runtime_lorentz(inputs.grid.dt, inputs.materials, lorentz_spec)
            carry_arrays.extend(
                (f"lorentz.{name}", arr)
                for name, arr in zip(initial_lorentz_state._fields, initial_lorentz_state)
            )
            replay_inputs.append("lorentz_spec")
            replay_outputs.append("final_lorentz_state")
        if boundary == "cpml":
            cpml_params, initial_cpml_state = init_cpml(inputs.grid, pec_faces=set(pec_faces))
            carry_arrays.extend(
                (f"cpml.{name}", arr)
                for name, arr in zip(initial_cpml_state._fields, initial_cpml_state)
            )
            replay_inputs.extend(("cpml_params", "pec_faces"))
            replay_outputs.append("final_cpml_state")
        ntff_box = inputs.ntff_box
        if ntff_box is not None:
            initial_ntff = init_ntff_data(ntff_box)
            carry_arrays.extend(
                (f"ntff.{name}", arr)
                for name, arr in zip(initial_ntff._fields, initial_ntff)
            )
            replay_inputs.append("ntff_box")
            replay_outputs.append("final_ntff_data")
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
            periodic=inputs.periodic,
            n_steps=inputs.n_steps,
            dt=inputs.grid.dt,
            dx=inputs.grid.dx,
            eps_r=inputs.materials.eps_r,
            mu_r=inputs.materials.mu_r,
            sigma=inputs.materials.sigma,
            debye_spec=debye_spec,
            lorentz_spec=lorentz_spec,
            cpml_axes=cpml_axes,
            pec_axes=inputs.pec_axes,
            pec_faces=pec_faces,
            cpml_params=cpml_params,
            ntff_box=ntff_box,
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
        if self.ntff_box is None:
            return phase1_forward_result(self.grid, self.run_time_series(eps_override))
        observables = _make_phase1_hybrid_observable_forward(self)(self.resolved_eps_r(eps_override))
        return phase1_forward_result(
            self.grid,
            observables.time_series,
            ntff_data=observables.ntff_data,
            ntff_box=self.ntff_box,
        )


@dataclass(frozen=True)
class Phase1HybridInputs:
    """Seam-owned input spec for Phase 1 hybrid preparation."""

    boundary: str
    periodic: tuple[bool, bool, bool]
    materials: MaterialArrays | None
    raw_sources: list[tuple[int, int, int, str, jnp.ndarray]] | tuple[tuple[int, int, int, str, jnp.ndarray], ...]
    probes: list[ProbeSpec] | tuple[ProbeSpec, ...]
    port_metadata: object | None
    debye_spec: tuple | None
    debye: tuple | None
    lorentz_spec: tuple | None
    lorentz: tuple | None
    ntff_box: object | None
    waveguide_ports: list | tuple | None
    pec_mask: jnp.ndarray | None
    pec_occupancy: jnp.ndarray | None
    grid: Grid | NonUniformGrid | None = None
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
            port_metadata=None,
            debye_spec=None,
            debye=None,
            lorentz_spec=None,
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
            port_metadata=prepared.port_metadata,
            debye_spec=prepared.debye_spec,
            debye=prepared.debye,
            lorentz_spec=prepared.lorentz_spec,
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
        grid: Grid | NonUniformGrid | None,
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
    port_metadata: object | None = None

    @classmethod
    def unsupported(
        cls,
        *,
        reasons: tuple[str, ...],
        source_count: int,
        probe_count: int,
        boundary: str,
        periodic: tuple[bool, bool, bool],
        port_metadata: object | None = None,
    ) -> Phase1HybridInspection:
        return cls(
            supported=False,
            reasons=reasons,
            inventory=None,
            source_count=source_count,
            probe_count=probe_count,
            boundary=boundary,
            periodic=periodic,
            port_metadata=port_metadata,
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
            lorentz_spec=inputs.lorentz_spec,
            lorentz=inputs.lorentz,
            ntff_box=inputs.ntff_box,
            waveguide_ports=inputs.waveguide_ports,
            port_metadata=inputs.port_metadata,
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
                port_metadata=inputs.port_metadata,
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
            port_metadata=None,
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


def _passive_lumped_port_cells(port_metadata: object | None) -> tuple[tuple[int, int, int], ...]:
    if port_metadata is None:
        return ()
    return tuple(
        tuple(int(v) for v in cell)
        for cell in getattr(port_metadata, "passive_lumped_port_cells", ())
    )


def _passive_lumped_port_sigmas(port_metadata: object | None) -> tuple[float, ...]:
    if port_metadata is None:
        return ()
    return tuple(float(value) for value in getattr(port_metadata, "passive_lumped_port_sigmas", ()))


def _passive_lumped_port_components(port_metadata: object | None) -> tuple[str, ...]:
    if port_metadata is None:
        return ()
    return tuple(str(value) for value in getattr(port_metadata, "passive_lumped_port_components", ()))


def _passive_lumped_port_had_pec(port_metadata: object | None) -> tuple[bool, ...]:
    if port_metadata is None:
        return ()
    return tuple(bool(value) for value in getattr(port_metadata, "passive_lumped_port_had_pec", ()))


def _supports_phase2_lumped_port_proxy_subset(port_metadata: object | None) -> bool:
    """Return whether metadata matches the bounded Phase II lumped-port subset."""

    if port_metadata is None:
        return False

    passive_cells = _passive_lumped_port_cells(port_metadata)
    passive_components = _passive_lumped_port_components(port_metadata)
    passive_sigmas = _passive_lumped_port_sigmas(port_metadata)
    passive_had_pec = _passive_lumped_port_had_pec(port_metadata)
    port_cells = (tuple(port_metadata.excited_lumped_port_cell),) if port_metadata.excited_lumped_port_cell else ()
    port_cells = port_cells + passive_cells

    return bool(
        port_metadata.total_ports == port_metadata.excited_ports + port_metadata.passive_ports
        and port_metadata.total_ports in {1, 2}
        and port_metadata.excited_ports == 1
        and port_metadata.passive_ports in {0, 1}
        and port_metadata.wire_ports == 0
        and port_metadata.waveguide_ports == 0
        and port_metadata.floquet_ports == 0
        and port_metadata.soft_source_count == 0
        and port_metadata.excited_lumped_port_cell is not None
        and port_metadata.excited_lumped_port_component is not None
        and port_metadata.excited_lumped_port_sigma is not None
        and not port_metadata.excited_port_had_pec
        and len(passive_cells) == port_metadata.passive_ports
        and len(passive_components) == port_metadata.passive_ports
        and len(passive_sigmas) == port_metadata.passive_ports
        and len(passive_had_pec) == port_metadata.passive_ports
        and not any(passive_had_pec)
        and len(set(port_cells)) == len(port_cells)
    )


def _supported_lumped_port_sigmas_by_cell(
    port_metadata: object | None,
) -> dict[tuple[int, int, int], float]:
    if not _supports_phase2_lumped_port_proxy_subset(port_metadata):
        return {}

    assert port_metadata is not None
    expected = {
        tuple(int(v) for v in port_metadata.excited_lumped_port_cell):
            float(port_metadata.excited_lumped_port_sigma)
    }
    for cell, sigma in zip(
        _passive_lumped_port_cells(port_metadata),
        _passive_lumped_port_sigmas(port_metadata),
    ):
        expected[cell] = float(sigma)
    return expected


def _sigma_matches_supported_lumped_port_subset(
    sigma: np.ndarray,
    port_metadata: object | None,
) -> bool:
    expected = _supported_lumped_port_sigmas_by_cell(port_metadata)
    if not expected:
        return False
    nonzero = np.argwhere(np.abs(sigma) > 0.0)
    if nonzero.shape[0] != len(expected):
        return False
    for raw_cell in nonzero:
        cell = tuple(int(v) for v in raw_cell)
        if cell not in expected:
            return False
        if not np.isclose(
            float(sigma[cell]),
            expected[cell],
            rtol=1e-6,
            atol=1e-8,
        ):
            return False
    return True


def _has_passive_lumped_port(port_metadata: object | None) -> bool:
    return bool(port_metadata is not None and getattr(port_metadata, "passive_ports", 0) > 0)


def _port_metadata_had_preexisting_pec(port_metadata: object | None) -> bool:
    return bool(
        port_metadata is not None
        and (
            getattr(port_metadata, "excited_port_had_pec", False)
            or any(_passive_lumped_port_had_pec(port_metadata))
        )
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
    lorentz_spec: tuple | None,
    lorentz: tuple | None,
    ntff_box: object | None,
    waveguide_ports: list | tuple | None,
    port_metadata: object | None = None,
    pec_mask: jnp.ndarray | None = None,
    pec_occupancy: jnp.ndarray | None = None,
    n_warmup: int = 0,
    checkpoint_every: int | None = None,
) -> tuple[str, ...]:
    """Return explicit reasons that the Phase 1 hybrid lane is unsupported."""

    reasons: list[str] = []
    if boundary not in {"pec", "cpml"}:
        reasons.append(f"boundary={boundary!r} is unsupported")
    if debye_spec is not None and lorentz_spec is not None:
        reasons.append("mixed Debye+Lorentz dispersion is unsupported")
    if debye is not None and debye_spec is None:
        reasons.append("Debye reconstruction metadata is unavailable")
    if lorentz is not None and lorentz_spec is None:
        reasons.append("Lorentz reconstruction metadata is unavailable")
    if lorentz_spec is not None:
        lorentz_poles, _ = lorentz_spec
        if any(getattr(pole, "omega_0", None) == 0.0 for pole in lorentz_poles):
            reasons.append("Drude-shaped Lorentz poles are unsupported")
    if port_metadata is not None and port_metadata.total_ports > 0:
        if port_metadata.soft_source_count > 0:
            reasons.append("mixed add_source()-style J-sources and port excitation are unsupported")
        if not _supports_phase2_lumped_port_proxy_subset(port_metadata):
            reasons.append(
                "only one excited lumped port with at most one passive lumped port is supported in Phase 2 hybrid mode"
            )
        if _port_metadata_had_preexisting_pec(port_metadata):
            reasons.append("pre-existing PEC at lumped-port cells is unsupported")
        if ntff_box is not None and _has_passive_lumped_port(port_metadata):
            reasons.append("NTFF with passive lumped-port proxy workflows is unsupported")
    if port_metadata is not None and getattr(port_metadata, "floquet_ports", 0) > 0:
        reasons.append("floquet periodic workflows are unsupported")
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
    if np.any(np.abs(sigma) > 0.0) and not _sigma_matches_supported_lumped_port_subset(sigma, port_metadata):
        reasons.append("lossy materials / port-loaded conductivity are unsupported")
    return tuple(dict.fromkeys(reasons))


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
    port_metadata: object | None,
    debye_spec: tuple | None,
    debye: tuple | None,
    lorentz_spec: tuple | None,
    lorentz: tuple | None,
    ntff_box: object | None,
    waveguide_ports: list | tuple | None,
    pec_mask: jnp.ndarray | None,
    pec_occupancy: jnp.ndarray | None,
    grid: Grid | NonUniformGrid | None = None,
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
        lorentz_spec=lorentz_spec,
        lorentz=lorentz,
        ntff_box=ntff_box,
        waveguide_ports=waveguide_ports,
        port_metadata=port_metadata,
        pec_mask=pec_mask,
        pec_occupancy=pec_occupancy,
        n_warmup=n_warmup,
        checkpoint_every=checkpoint_every,
    )
    if isinstance(grid, NonUniformGrid):
        cpml = grid.cpml_layers
        dx_phys = np.asarray(grid.dx_arr)[cpml : grid.nx - cpml]
        dy_phys = np.asarray(grid.dy_arr)[cpml : grid.ny - cpml]
        if periodic != (False, False, False):
            reasons = tuple(
                dict.fromkeys(
                    reasons + ("Phase V does not support combined non-uniform + periodic hybrid workflows",)
                )
            )
        if (
            not np.allclose(dx_phys, dx_phys[0])
            or not np.allclose(dy_phys, dy_phys[0])
        ):
            reasons = tuple(dict.fromkeys(reasons + ("non-uniform grids are unsupported",)))
        if ntff_box is not None:
            reasons = tuple(
                dict.fromkeys(reasons + ("Phase V nonuniform hybrid does not support NTFF",))
            )
        if debye_spec is not None or lorentz_spec is not None:
            reasons = tuple(
                dict.fromkeys(
                    reasons + ("Phase V nonuniform hybrid supports only lossless nondispersive materials",)
                )
            )
        if np.any(np.abs(np.asarray(materials.sigma)) > 0.0):
            reasons = tuple(
                dict.fromkeys(
                    reasons + ("Phase V nonuniform hybrid supports only zero sigma materials",)
                )
            )
    elif periodic != (False, False, False):
        if ntff_box is not None:
            reasons = tuple(
                dict.fromkeys(reasons + ("Phase V bounded periodic hybrid does not support NTFF",))
            )
        if debye_spec is not None or lorentz_spec is not None:
            reasons = tuple(
                dict.fromkeys(
                    reasons + ("Phase V bounded periodic hybrid supports only lossless nondispersive materials",)
                )
            )
        if np.any(np.abs(np.asarray(materials.sigma)) > 0.0):
            reasons = tuple(
                dict.fromkeys(
                    reasons + ("Phase V bounded periodic hybrid supports only zero sigma materials",)
                )
            )
        if port_metadata is not None and (
            getattr(port_metadata, "total_ports", 0) > 0
            or getattr(port_metadata, "floquet_ports", 0) > 0
        ):
            reasons = tuple(
                dict.fromkeys(
                    reasons + ("Phase V bounded periodic hybrid supports only add_source()/probe workflows",)
                )
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
                port_metadata=port_metadata,
                debye_spec=debye_spec,
                debye=debye,
                lorentz_spec=lorentz_spec,
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
        port_metadata=port_metadata,
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
    lorentz_spec: tuple | None = None,
    ntff_box: object | None = None,
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
            port_metadata=None,
            debye_spec=debye_spec,
            debye=None,
            lorentz_spec=lorentz_spec,
            lorentz=None,
            ntff_box=ntff_box,
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
    if isinstance(context.grid, NonUniformGrid):
        if context.debye_spec is not None or context.lorentz_spec is not None:
            raise ValueError("mixed or dispersive non-uniform hybrid support is unsupported")
        sources = [
            (i, j, k, component, src_waveforms[:, idx])
            for idx, (i, j, k, component) in enumerate(context.src_meta)
        ]
        probes = list(context.prb_meta)
        result = run_nonuniform(
            context.grid,
            materials,
            context.n_steps,
            sources=sources,
            probes=probes,
            checkpoint=True,
            emit_time_series=True,
        )
        return result["time_series"]
    if context.periodic != (False, False, False) and (
        context.debye_spec is not None or context.lorentz_spec is not None
    ):
        raise ValueError("bounded periodic hybrid supports only lossless nondispersive materials")
    if context.debye_spec is not None and context.lorentz_spec is not None:
        raise ValueError("mixed Debye+Lorentz dispersion is unsupported")
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
    if context.lorentz_spec is not None:
        lorentz_coeffs, lorentz_state = _runtime_lorentz(context.dt, materials, context.lorentz_spec)
        if context.boundary == "cpml":
            _, _, _, time_series = _run_uniform_lorentz_cpml_time_series(
                context,
                materials,
                lorentz_coeffs,
                lorentz_state,
                src_waveforms,
            )
            return time_series
        if context.boundary == "pec":
            _, _, time_series = _run_uniform_lorentz_pec_time_series(
                context,
                materials,
                lorentz_coeffs,
                lorentz_state,
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
    _, time_series = _run_uniform_lossless_pec_time_series(
        context,
        coeffs,
        src_waveforms,
        materials=materials,
    )
    return time_series


def make_phase1_hybrid_forward(context: Phase1HybridContext) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create a custom_vjp forward op for the Phase 1 hybrid seam."""

    if isinstance(context.grid, NonUniformGrid):
        def _forward_time_series(eps_r: jnp.ndarray) -> jnp.ndarray:
            return run_phase1_forward_time_series(context, eps_r)

        return _forward_time_series

    @jax.custom_vjp
    def _forward_time_series(eps_r: jnp.ndarray) -> jnp.ndarray:
        return run_phase1_forward_time_series(context, eps_r)

    def _forward_fwd(eps_r: jnp.ndarray):
        materials = _materials_from_eps(context, eps_r)
        src_waveforms = _source_waveforms_from_eps(context, eps_r)
        if context.debye_spec is not None and context.lorentz_spec is not None:
            raise ValueError("mixed Debye+Lorentz dispersion is unsupported")
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
        if context.lorentz_spec is not None:
            lorentz_coeffs, lorentz_state = _runtime_lorentz(context.dt, materials, context.lorentz_spec)
            if context.boundary == "cpml":
                _, _, _, time_series, states_before, cpml_states_before, lorentz_states_before = (
                    _run_uniform_lorentz_cpml_time_series(
                        context,
                        materials,
                        lorentz_coeffs,
                        lorentz_state,
                        src_waveforms,
                        return_trace=True,
                    )
                )
                return time_series, (
                    eps_r,
                    states_before,
                    cpml_states_before,
                    lorentz_states_before,
                    context.src_waveforms_raw,
                )
            if context.boundary == "pec":
                _, _, time_series, states_before, lorentz_states_before = _run_uniform_lorentz_pec_time_series(
                    context,
                    materials,
                    lorentz_coeffs,
                    lorentz_state,
                    src_waveforms,
                    return_trace=True,
                )
                return time_series, (eps_r, states_before, lorentz_states_before, context.src_waveforms_raw)
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
        if context.lorentz_spec is not None and context.boundary == "cpml":
            eps_r, states_before, cpml_states_before, lorentz_states_before, raw_src_waveforms = res
            zero_state = _zeros_like_state(context.initial_state)
            zero_cpml = _zero_cpml_state(context.grid)
            zero_lorentz = _zero_lorentz_state(
                _runtime_lorentz(context.dt, _materials_from_eps(context, eps_r), context.lorentz_spec)[1]
            )
            zero_eps = jnp.zeros_like(eps_r)

            def reverse_step(carry, xs):
                lambda_next, lambda_cpml_next, lambda_lorentz_next, grad_eps = carry
                state_before, cpml_before, lorentz_before, raw_src_vals, probe_cot = xs

                def step_from_eps(state, cpml_state, lorentz_state, eps_local):
                    materials_local = _materials_from_eps(context, eps_local)
                    lorentz_coeffs, _ = _runtime_lorentz(
                        context.dt,
                        materials_local,
                        context.lorentz_spec,
                    )
                    return _uniform_lorentz_cpml_step(
                        state,
                        cpml_state,
                        lorentz_state,
                        _source_values_from_eps(context, eps_local, raw_src_vals),
                        materials_local,
                        lorentz_coeffs,
                        context.grid,
                        context.cpml_params,
                        context.cpml_axes,
                        context.pec_axes,
                        context.pec_faces,
                        context.src_meta,
                        context.prb_meta,
                    )

                _, step_vjp = jax.vjp(step_from_eps, state_before, cpml_before, lorentz_before, eps_r)
                lambda_before, lambda_cpml_before, lambda_lorentz_before, grad_eps_step = step_vjp(
                    (lambda_next, lambda_cpml_next, lambda_lorentz_next, probe_cot)
                )
                return (
                    lambda_before,
                    lambda_cpml_before,
                    lambda_lorentz_before,
                    grad_eps + grad_eps_step,
                ), None

            (_, _, _, grad_eps), _ = jax.lax.scan(
                reverse_step,
                (zero_state, zero_cpml, zero_lorentz, zero_eps),
                (states_before, cpml_states_before, lorentz_states_before, raw_src_waveforms, probe_bar),
                reverse=True,
            )
            return (grad_eps,)

        if context.lorentz_spec is not None and context.boundary == "pec":
            eps_r, states_before, lorentz_states_before, raw_src_waveforms = res
            zero_state = _zeros_like_state(context.initial_state)
            zero_lorentz = _zero_lorentz_state(
                _runtime_lorentz(context.dt, _materials_from_eps(context, eps_r), context.lorentz_spec)[1]
            )
            zero_eps = jnp.zeros_like(eps_r)

            def reverse_step(carry, xs):
                lambda_next, lambda_lorentz_next, grad_eps = carry
                state_before, lorentz_before, raw_src_vals, probe_cot = xs

                def step_from_eps(state, lorentz_state, eps_local):
                    materials_local = _materials_from_eps(context, eps_local)
                    lorentz_coeffs, _ = _runtime_lorentz(
                        context.dt,
                        materials_local,
                        context.lorentz_spec,
                    )
                    return _uniform_lorentz_pec_step(
                        state,
                        lorentz_state,
                        _source_values_from_eps(context, eps_local, raw_src_vals),
                        materials_local,
                        lorentz_coeffs,
                        context.grid,
                        context.pec_axes,
                        context.pec_faces,
                        context.src_meta,
                        context.prb_meta,
                    )

                _, step_vjp = jax.vjp(step_from_eps, state_before, lorentz_before, eps_r)
                lambda_before, lambda_lorentz_before, grad_eps_step = step_vjp(
                    (lambda_next, lambda_lorentz_next, probe_cot)
                )
                return (lambda_before, lambda_lorentz_before, grad_eps + grad_eps_step), None

            (_, _, grad_eps), _ = jax.lax.scan(
                reverse_step,
                (zero_state, zero_lorentz, zero_eps),
                (states_before, lorentz_states_before, raw_src_waveforms, probe_bar),
                reverse=True,
            )
            return (grad_eps,)

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
                        periodic=context.periodic,
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
                    materials=_materials_from_eps(context, eps_local),
                    grid=context.grid if isinstance(context.grid, Grid) else None,
                    periodic=context.periodic,
                    pec_axes=context.pec_axes,
                    pec_faces=context.pec_faces,
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


def _run_phase1_forward_observables(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
) -> _Phase1HybridObservables:
    materials = _materials_from_eps(context, eps_r)
    src_waveforms = _source_waveforms_from_eps(context, eps_r)
    _, _, _, observables = _run_phase1_trace_for_observables(context, materials, src_waveforms)
    return observables


def _run_phase1_trace_for_observables(
    context: Phase1HybridContext,
    materials: MaterialArrays,
    src_waveforms: jnp.ndarray,
):
    if context.debye_spec is not None and context.lorentz_spec is not None:
        raise ValueError("mixed Debye+Lorentz dispersion is unsupported")

    if context.debye_spec is not None:
        debye_coeffs, debye_state = _runtime_debye(context.dt, materials, context.debye_spec)
        if context.boundary == "cpml":
            (
                final_state,
                _final_cpml_state,
                _final_debye_state,
                time_series,
                states_before,
                cpml_states_before,
                debye_states_before,
            ) = _run_uniform_debye_cpml_time_series(
                context,
                materials,
                debye_coeffs,
                debye_state,
                src_waveforms,
                return_trace=True,
            )
            aux_states_before = (cpml_states_before, debye_states_before)
        elif context.boundary == "pec":
            final_state, _final_debye_state, time_series, states_before, debye_states_before = (
                _run_uniform_debye_pec_time_series(
                    context,
                    materials,
                    debye_coeffs,
                    debye_state,
                    src_waveforms,
                    return_trace=True,
                )
            )
            aux_states_before = debye_states_before
        else:
            raise ValueError(f"boundary={context.boundary!r} is unsupported")
    elif context.lorentz_spec is not None:
        lorentz_coeffs, lorentz_state = _runtime_lorentz(context.dt, materials, context.lorentz_spec)
        if context.boundary == "cpml":
            (
                final_state,
                _final_cpml_state,
                _final_lorentz_state,
                time_series,
                states_before,
                cpml_states_before,
                lorentz_states_before,
            ) = _run_uniform_lorentz_cpml_time_series(
                context,
                materials,
                lorentz_coeffs,
                lorentz_state,
                src_waveforms,
                return_trace=True,
            )
            aux_states_before = (cpml_states_before, lorentz_states_before)
        elif context.boundary == "pec":
            final_state, _final_lorentz_state, time_series, states_before, lorentz_states_before = (
                _run_uniform_lorentz_pec_time_series(
                    context,
                    materials,
                    lorentz_coeffs,
                    lorentz_state,
                    src_waveforms,
                    return_trace=True,
                )
            )
            aux_states_before = lorentz_states_before
        else:
            raise ValueError(f"boundary={context.boundary!r} is unsupported")
    elif context.boundary == "cpml":
        final_state, _final_cpml_state, time_series, states_before, cpml_states_before = (
            _run_uniform_lossless_cpml_time_series(
                context,
                materials,
                src_waveforms,
                return_trace=True,
            )
        )
        aux_states_before = cpml_states_before
    elif context.boundary == "pec":
        final_state, time_series, states_before = _run_uniform_lossless_pec_time_series(
            context,
            _coeffs_from_materials(context, materials),
            src_waveforms,
            return_trace=True,
        )
        aux_states_before = None
    else:
        raise ValueError(f"boundary={context.boundary!r} is unsupported")

    ntff_data = None
    if context.ntff_box is not None:
        ntff_data = _accumulate_ntff_from_post_states(
            context,
            _states_after_from_trace(states_before, final_state),
        )

    return final_state, states_before, aux_states_before, _Phase1HybridObservables(
        time_series=time_series,
        ntff_data=ntff_data,
    )


def _reverse_phase1_trace_from_observables(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
    states_before: Phase1FieldState,
    aux_states_before,
    raw_src_waveforms: jnp.ndarray,
    time_series_bar: jnp.ndarray,
    post_state_bars: Phase1FieldState,
) -> jnp.ndarray:
    if context.debye_spec is not None and context.boundary == "cpml":
        cpml_states_before, debye_states_before = aux_states_before
        zero_state = _zeros_like_state(context.initial_state)
        zero_cpml = _zero_cpml_state(context.grid)
        zero_debye = _zero_debye_state(
            _runtime_debye(context.dt, _materials_from_eps(context, eps_r), context.debye_spec)[1]
        )
        zero_eps = jnp.zeros_like(eps_r)

        def reverse_step(carry, xs):
            lambda_next, lambda_cpml_next, lambda_debye_next, grad_eps = carry
            state_before, cpml_before, debye_before, raw_src_vals, probe_cot, state_bar = xs

            def step_from_eps(state, cpml_state, debye_state, eps_local):
                materials_local = _materials_from_eps(context, eps_local)
                debye_coeffs, _ = _runtime_debye(context.dt, materials_local, context.debye_spec)
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
                (_add_field_states(lambda_next, state_bar), lambda_cpml_next, lambda_debye_next, probe_cot)
            )
            return (lambda_before, lambda_cpml_before, lambda_debye_before, grad_eps + grad_eps_step), None

        (_, _, _, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (zero_state, zero_cpml, zero_debye, zero_eps),
            (states_before, cpml_states_before, debye_states_before, raw_src_waveforms, time_series_bar, post_state_bars),
            reverse=True,
        )
        return grad_eps

    if context.debye_spec is not None and context.boundary == "pec":
        debye_states_before = aux_states_before
        zero_state = _zeros_like_state(context.initial_state)
        zero_debye = _zero_debye_state(
            _runtime_debye(context.dt, _materials_from_eps(context, eps_r), context.debye_spec)[1]
        )
        zero_eps = jnp.zeros_like(eps_r)

        def reverse_step(carry, xs):
            lambda_next, lambda_debye_next, grad_eps = carry
            state_before, debye_before, raw_src_vals, probe_cot, state_bar = xs

            def step_from_eps(state, debye_state, eps_local):
                materials_local = _materials_from_eps(context, eps_local)
                debye_coeffs, _ = _runtime_debye(context.dt, materials_local, context.debye_spec)
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
                (_add_field_states(lambda_next, state_bar), lambda_debye_next, probe_cot)
            )
            return (lambda_before, lambda_debye_before, grad_eps + grad_eps_step), None

        (_, _, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (zero_state, zero_debye, zero_eps),
            (states_before, debye_states_before, raw_src_waveforms, time_series_bar, post_state_bars),
            reverse=True,
        )
        return grad_eps

    if context.lorentz_spec is not None and context.boundary == "cpml":
        cpml_states_before, lorentz_states_before = aux_states_before
        zero_state = _zeros_like_state(context.initial_state)
        zero_cpml = _zero_cpml_state(context.grid)
        zero_lorentz = _zero_lorentz_state(
            _runtime_lorentz(context.dt, _materials_from_eps(context, eps_r), context.lorentz_spec)[1]
        )
        zero_eps = jnp.zeros_like(eps_r)

        def reverse_step(carry, xs):
            lambda_next, lambda_cpml_next, lambda_lorentz_next, grad_eps = carry
            state_before, cpml_before, lorentz_before, raw_src_vals, probe_cot, state_bar = xs

            def step_from_eps(state, cpml_state, lorentz_state, eps_local):
                materials_local = _materials_from_eps(context, eps_local)
                lorentz_coeffs, _ = _runtime_lorentz(context.dt, materials_local, context.lorentz_spec)
                return _uniform_lorentz_cpml_step(
                    state,
                    cpml_state,
                    lorentz_state,
                    _source_values_from_eps(context, eps_local, raw_src_vals),
                    materials_local,
                    lorentz_coeffs,
                    context.grid,
                    context.cpml_params,
                    context.cpml_axes,
                    context.pec_axes,
                    context.pec_faces,
                    context.src_meta,
                    context.prb_meta,
                )

            _, step_vjp = jax.vjp(step_from_eps, state_before, cpml_before, lorentz_before, eps_r)
            lambda_before, lambda_cpml_before, lambda_lorentz_before, grad_eps_step = step_vjp(
                (_add_field_states(lambda_next, state_bar), lambda_cpml_next, lambda_lorentz_next, probe_cot)
            )
            return (lambda_before, lambda_cpml_before, lambda_lorentz_before, grad_eps + grad_eps_step), None

        (_, _, _, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (zero_state, zero_cpml, zero_lorentz, zero_eps),
            (states_before, cpml_states_before, lorentz_states_before, raw_src_waveforms, time_series_bar, post_state_bars),
            reverse=True,
        )
        return grad_eps

    if context.lorentz_spec is not None and context.boundary == "pec":
        lorentz_states_before = aux_states_before
        zero_state = _zeros_like_state(context.initial_state)
        zero_lorentz = _zero_lorentz_state(
            _runtime_lorentz(context.dt, _materials_from_eps(context, eps_r), context.lorentz_spec)[1]
        )
        zero_eps = jnp.zeros_like(eps_r)

        def reverse_step(carry, xs):
            lambda_next, lambda_lorentz_next, grad_eps = carry
            state_before, lorentz_before, raw_src_vals, probe_cot, state_bar = xs

            def step_from_eps(state, lorentz_state, eps_local):
                materials_local = _materials_from_eps(context, eps_local)
                lorentz_coeffs, _ = _runtime_lorentz(context.dt, materials_local, context.lorentz_spec)
                return _uniform_lorentz_pec_step(
                    state,
                    lorentz_state,
                    _source_values_from_eps(context, eps_local, raw_src_vals),
                    materials_local,
                    lorentz_coeffs,
                    context.grid,
                    context.pec_axes,
                    context.pec_faces,
                    context.src_meta,
                    context.prb_meta,
                )

            _, step_vjp = jax.vjp(step_from_eps, state_before, lorentz_before, eps_r)
            lambda_before, lambda_lorentz_before, grad_eps_step = step_vjp(
                (_add_field_states(lambda_next, state_bar), lambda_lorentz_next, probe_cot)
            )
            return (lambda_before, lambda_lorentz_before, grad_eps + grad_eps_step), None

        (_, _, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (zero_state, zero_lorentz, zero_eps),
            (states_before, lorentz_states_before, raw_src_waveforms, time_series_bar, post_state_bars),
            reverse=True,
        )
        return grad_eps

    if context.boundary == "cpml":
        cpml_states_before = aux_states_before
        zero_state = _zeros_like_state(context.initial_state)
        zero_cpml = _zero_cpml_state(context.grid)
        zero_eps = jnp.zeros_like(eps_r)

        def reverse_step(carry, xs):
            lambda_next, lambda_cpml_next, grad_eps = carry
            state_before, cpml_before, raw_src_vals, probe_cot, state_bar = xs

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
                    periodic=context.periodic,
                )

            _, step_vjp = jax.vjp(step_from_eps, state_before, cpml_before, eps_r)
            lambda_before, lambda_cpml_before, grad_eps_step = step_vjp(
                (_add_field_states(lambda_next, state_bar), lambda_cpml_next, probe_cot)
            )
            return (lambda_before, lambda_cpml_before, grad_eps + grad_eps_step), None

        (_, _, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (zero_state, zero_cpml, zero_eps),
            (states_before, cpml_states_before, raw_src_waveforms, time_series_bar, post_state_bars),
            reverse=True,
        )
        return grad_eps

    zero_state = _zeros_like_state(context.initial_state)
    zero_eps = jnp.zeros_like(eps_r)

    def reverse_step(carry, xs):
        lambda_next, grad_eps = carry
        state_before, raw_src_vals, probe_cot, state_bar = xs

        def step_from_eps(state, eps_local):
            return _uniform_lossless_pec_step(
                state,
                _source_values_from_eps(context, eps_local, raw_src_vals),
                _coeffs_from_eps(context, eps_local),
                context.src_meta,
                context.prb_meta,
                materials=_materials_from_eps(context, eps_local),
                grid=context.grid if isinstance(context.grid, Grid) else None,
                periodic=context.periodic,
                pec_axes=context.pec_axes,
                pec_faces=context.pec_faces,
            )

        _, step_vjp = jax.vjp(step_from_eps, state_before, eps_r)
        lambda_before, grad_eps_step = step_vjp((_add_field_states(lambda_next, state_bar), probe_cot))
        return (lambda_before, grad_eps + grad_eps_step), None

    (_, grad_eps), _ = jax.lax.scan(
        reverse_step,
        (zero_state, zero_eps),
        (states_before, raw_src_waveforms, time_series_bar, post_state_bars),
        reverse=True,
    )
    return grad_eps



def _phase3_strategy_b_segments(n_steps: int, checkpoint_every: int) -> tuple[tuple[int, int], ...]:
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive for Strategy B")
    return tuple(
        (start, min(start + checkpoint_every, n_steps))
        for start in range(0, n_steps, checkpoint_every)
    )


def _stack_phase1_field_states(states: tuple[Phase1FieldState, ...]) -> Phase1FieldState:
    return Phase1FieldState(
        *(jnp.stack([state_field for state_field in fields], axis=0) for fields in zip(*states))
    )


def _phase1_field_state_at(states: Phase1FieldState, index: int) -> Phase1FieldState:
    return Phase1FieldState(*(field[index] for field in states))


def _stack_cpml_states(states: tuple[CPMLState, ...]) -> CPMLState:
    return CPMLState(
        *(jnp.stack([state_field for state_field in fields], axis=0) for fields in zip(*states))
    )


def _cpml_state_at(states: CPMLState, index: int) -> CPMLState:
    return CPMLState(*(field[index] for field in states))


def _run_phase3_strategy_b_lossless_pec_forward(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
    checkpoint_every: int,
) -> tuple[jnp.ndarray, Phase1FieldState]:
    coeffs = _coeffs_from_eps(context, eps_r)
    src_waveforms = _source_waveforms_from_eps(context, eps_r)
    state = context.initial_state
    checkpoint_states: list[Phase1FieldState] = []
    time_series_segments: list[jnp.ndarray] = []

    for start, end in _phase3_strategy_b_segments(context.n_steps, checkpoint_every):
        checkpoint_states.append(state)
        state, segment_time_series = _run_uniform_lossless_pec_time_series(
            context,
            coeffs,
            src_waveforms[start:end],
            initial_state=state,
        )
        time_series_segments.append(segment_time_series)

    return jnp.concatenate(time_series_segments, axis=0), _stack_phase1_field_states(
        tuple(checkpoint_states)
    )


def _run_phase3_strategy_b_lossless_cpml_forward(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
    checkpoint_every: int,
) -> tuple[jnp.ndarray, Phase1FieldState, CPMLState]:
    materials = _materials_from_eps(context, eps_r)
    src_waveforms = _source_waveforms_from_eps(context, eps_r)
    state = context.initial_state
    cpml_state = _zero_cpml_state(context.grid)
    checkpoint_states: list[Phase1FieldState] = []
    checkpoint_cpml_states: list[CPMLState] = []
    time_series_segments: list[jnp.ndarray] = []

    for start, end in _phase3_strategy_b_segments(context.n_steps, checkpoint_every):
        checkpoint_states.append(state)
        checkpoint_cpml_states.append(cpml_state)
        state, cpml_state, segment_time_series = _run_uniform_lossless_cpml_time_series(
            context,
            materials,
            src_waveforms[start:end],
            initial_state=state,
            initial_cpml_state=cpml_state,
        )
        time_series_segments.append(segment_time_series)

    return (
        jnp.concatenate(time_series_segments, axis=0),
        _stack_phase1_field_states(tuple(checkpoint_states)),
        _stack_cpml_states(tuple(checkpoint_cpml_states)),
    )


def _reverse_phase3_strategy_b_lossless_pec(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
    checkpoints: Phase1FieldState,
    raw_src_waveforms: jnp.ndarray,
    probe_bar: jnp.ndarray,
    checkpoint_every: int,
) -> jnp.ndarray:
    coeffs = _coeffs_from_eps(context, eps_r)
    src_waveforms = _source_waveforms_from_eps(context, eps_r)
    lambda_next = _zeros_like_state(context.initial_state)
    grad_eps = jnp.zeros_like(eps_r)

    for segment_index, (start, end) in reversed(
        tuple(enumerate(_phase3_strategy_b_segments(context.n_steps, checkpoint_every)))
    ):
        segment_start_state = _phase1_field_state_at(checkpoints, segment_index)
        _, _, states_before = _run_uniform_lossless_pec_time_series(
            context,
            coeffs,
            src_waveforms[start:end],
            return_trace=True,
            initial_state=segment_start_state,
        )

        def reverse_step(carry, xs):
            lambda_after, grad_accum = carry
            state_before, raw_src_vals, probe_cot = xs

            def step_from_eps(state, eps_local):
                return _uniform_lossless_pec_step(
                    state,
                    _source_values_from_eps(context, eps_local, raw_src_vals),
                    _coeffs_from_eps(context, eps_local),
                    context.src_meta,
                    context.prb_meta,
                    materials=_materials_from_eps(context, eps_local),
                    grid=context.grid if isinstance(context.grid, Grid) else None,
                    periodic=context.periodic,
                    pec_axes=context.pec_axes,
                    pec_faces=context.pec_faces,
                )

            _, step_vjp = jax.vjp(step_from_eps, state_before, eps_r)
            lambda_before, grad_eps_step = step_vjp((lambda_after, probe_cot))
            return (lambda_before, grad_accum + grad_eps_step), None

        (lambda_next, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (lambda_next, grad_eps),
            (states_before, raw_src_waveforms[start:end], probe_bar[start:end]),
            reverse=True,
        )

    return grad_eps


def _reverse_phase3_strategy_b_lossless_cpml(
    context: Phase1HybridContext,
    eps_r: jnp.ndarray,
    checkpoints: Phase1FieldState,
    cpml_checkpoints: CPMLState,
    raw_src_waveforms: jnp.ndarray,
    probe_bar: jnp.ndarray,
    checkpoint_every: int,
) -> jnp.ndarray:
    materials = _materials_from_eps(context, eps_r)
    src_waveforms = _source_waveforms_from_eps(context, eps_r)
    lambda_next = _zeros_like_state(context.initial_state)
    lambda_cpml_next = _zero_cpml_state(context.grid)
    grad_eps = jnp.zeros_like(eps_r)

    for segment_index, (start, end) in reversed(
        tuple(enumerate(_phase3_strategy_b_segments(context.n_steps, checkpoint_every)))
    ):
        segment_start_state = _phase1_field_state_at(checkpoints, segment_index)
        segment_start_cpml = _cpml_state_at(cpml_checkpoints, segment_index)
        _, _, _, states_before, cpml_states_before = _run_uniform_lossless_cpml_time_series(
            context,
            materials,
            src_waveforms[start:end],
            return_trace=True,
            initial_state=segment_start_state,
            initial_cpml_state=segment_start_cpml,
        )

        def reverse_step(carry, xs):
            lambda_after, lambda_cpml_after, grad_accum = carry
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
                    periodic=context.periodic,
                )

            _, step_vjp = jax.vjp(step_from_eps, state_before, cpml_before, eps_r)
            lambda_before, lambda_cpml_before, grad_eps_step = step_vjp(
                (lambda_after, lambda_cpml_after, probe_cot)
            )
            return (lambda_before, lambda_cpml_before, grad_accum + grad_eps_step), None

        (lambda_next, lambda_cpml_next, grad_eps), _ = jax.lax.scan(
            reverse_step,
            (lambda_next, lambda_cpml_next, grad_eps),
            (
                states_before,
                cpml_states_before,
                raw_src_waveforms[start:end],
                probe_bar[start:end],
            ),
            reverse=True,
        )

    return grad_eps


def _make_phase3_strategy_b_source_probe_forward(
    context: Phase1HybridContext,
    checkpoint_every: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create the narrow Phase III Strategy B source/probe custom VJP."""

    @jax.custom_vjp
    def _forward_time_series(eps_r: jnp.ndarray) -> jnp.ndarray:
        if context.boundary == "cpml":
            time_series, _, _ = _run_phase3_strategy_b_lossless_cpml_forward(
                context,
                eps_r,
                checkpoint_every,
            )
        else:
            time_series, _ = _run_phase3_strategy_b_lossless_pec_forward(
                context,
                eps_r,
                checkpoint_every,
            )
        return time_series

    def _forward_fwd(eps_r: jnp.ndarray):
        if context.boundary == "cpml":
            time_series, checkpoints, cpml_checkpoints = (
                _run_phase3_strategy_b_lossless_cpml_forward(
                    context,
                    eps_r,
                    checkpoint_every,
                )
            )
            return time_series, (
                eps_r,
                checkpoints,
                cpml_checkpoints,
                context.src_waveforms_raw,
            )
        time_series, checkpoints = _run_phase3_strategy_b_lossless_pec_forward(
            context,
            eps_r,
            checkpoint_every,
        )
        return time_series, (eps_r, checkpoints, None, context.src_waveforms_raw)

    def _forward_bwd(res, probe_bar: jnp.ndarray):
        eps_r, checkpoints, cpml_checkpoints, raw_src_waveforms = res
        if context.boundary == "cpml":
            assert cpml_checkpoints is not None
            grad_eps = _reverse_phase3_strategy_b_lossless_cpml(
                context,
                eps_r,
                checkpoints,
                cpml_checkpoints,
                raw_src_waveforms,
                probe_bar,
                checkpoint_every,
            )
        else:
            grad_eps = _reverse_phase3_strategy_b_lossless_pec(
                context,
                eps_r,
                checkpoints,
                raw_src_waveforms,
                probe_bar,
                checkpoint_every,
            )
        return (grad_eps,)

    _forward_time_series.defvjp(_forward_fwd, _forward_bwd)
    return _forward_time_series


def _states_after_from_trace(
    states_before: Phase1FieldState,
    final_state: Phase1FieldState,
) -> Phase1FieldState:
    return Phase1FieldState(
        *(
            jnp.concatenate([field_trace[1:], final_field[None]], axis=0)
            for field_trace, final_field in zip(states_before, final_state)
        )
    )


def _accumulate_ntff_from_post_states(
    context: Phase1HybridContext,
    post_states: Phase1FieldState,
) -> NTFFData:
    assert context.ntff_box is not None

    def body(ntff_state, xs):
        step_idx, post_state = xs
        next_ntff = accumulate_ntff(
            ntff_state,
            _to_fdtd(post_state),
            context.ntff_box,
            context.dt,
            step_idx,
        )
        return next_ntff, None

    final_ntff, _ = jax.lax.scan(
        body,
        init_ntff_data(context.ntff_box),
        (jnp.arange(context.n_steps, dtype=jnp.int32), post_states),
    )
    return final_ntff


def _ntff_post_state_cotangents(
    context: Phase1HybridContext,
    states_before: Phase1FieldState,
    final_state: Phase1FieldState,
    ntff_state_bar: NTFFData,
) -> Phase1FieldState:
    post_states = _states_after_from_trace(states_before, final_state)
    _, vjp = jax.vjp(
        lambda stacked_states: _accumulate_ntff_from_post_states(context, stacked_states),
        post_states,
    )
    (post_state_bars,) = vjp(ntff_state_bar)
    return post_state_bars


def _zeros_like_state_trace(states: Phase1FieldState) -> Phase1FieldState:
    return Phase1FieldState(*(jnp.zeros_like(field) for field in states))


def _make_phase1_hybrid_observable_forward(
    context: Phase1HybridContext,
) -> Callable[[jnp.ndarray], _Phase1HybridObservables]:
    """Create a custom_vjp forward op for hybrid observable bundles."""

    @jax.custom_vjp
    def _forward_observables(eps_r: jnp.ndarray) -> _Phase1HybridObservables:
        return _run_phase1_forward_observables(context, eps_r)

    def _forward_fwd(eps_r: jnp.ndarray):
        materials = _materials_from_eps(context, eps_r)
        src_waveforms = _source_waveforms_from_eps(context, eps_r)
        final_state, states_before, aux_states_before, observables = _run_phase1_trace_for_observables(
            context,
            materials,
            src_waveforms,
        )
        return observables, (eps_r, final_state, states_before, aux_states_before, context.src_waveforms_raw)

    def _forward_bwd(res, observables_bar: _Phase1HybridObservables):
        eps_r, final_state, states_before, aux_states_before, raw_src_waveforms = res
        ntff_state_bar = observables_bar.ntff_data
        post_state_bars = (
            _ntff_post_state_cotangents(context, states_before, final_state, ntff_state_bar)
            if context.ntff_box is not None and ntff_state_bar is not None
            else None
        )
        grad_eps = _reverse_phase1_trace_from_observables(
            context,
            eps_r,
            states_before,
            aux_states_before,
            raw_src_waveforms,
            observables_bar.time_series,
            post_state_bars,
        )
        return (grad_eps,)

    _forward_observables.defvjp(_forward_fwd, _forward_bwd)
    return _forward_observables


def _run_uniform_lossless_pec_time_series(
    context: Phase1HybridContext,
    coeffs: UpdateCoeffs,
    src_waveforms: jnp.ndarray,
    *,
    materials: MaterialArrays | None = None,
    return_trace: bool = False,
    initial_state: Phase1FieldState | None = None,
):
    """Run the extracted time-series seam with precomputed coefficients."""

    def body(state, src_vals):
        next_state, probe_out = _uniform_lossless_pec_step(
            state,
            src_vals,
            coeffs,
            context.src_meta,
            context.prb_meta,
            materials=materials,
            grid=context.grid if isinstance(context.grid, Grid) else None,
            periodic=context.periodic,
            pec_axes=context.pec_axes,
            pec_faces=context.pec_faces,
        )
        if return_trace:
            return next_state, (probe_out, state)
        return next_state, probe_out

    start_state = context.initial_state if initial_state is None else initial_state
    final_state, outputs = jax.lax.scan(body, start_state, src_waveforms)
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
    initial_state: Phase1FieldState | None = None,
    initial_cpml_state: CPMLState | None = None,
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
            periodic=context.periodic,
        )
        if return_trace:
            return (next_state, next_cpml_state), (probe_out, state, cpml_state)
        return (next_state, next_cpml_state), probe_out

    (final_state, final_cpml_state), outputs = jax.lax.scan(
        body,
        (
            context.initial_state if initial_state is None else initial_state,
            _zero_cpml_state(context.grid) if initial_cpml_state is None else initial_cpml_state,
        ),
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


def _run_uniform_lorentz_pec_time_series(
    context: Phase1HybridContext,
    materials: MaterialArrays,
    lorentz_coeffs: LorentzCoeffs,
    lorentz_state: LorentzState,
    src_waveforms: jnp.ndarray,
    *,
    return_trace: bool = False,
):
    """Run the extracted time-series seam with Lorentz and PEC enforcement."""

    def body(carry, src_vals):
        state, lorentz_state_local = carry
        next_state, next_lorentz_state, probe_out = _uniform_lorentz_pec_step(
            state,
            lorentz_state_local,
            src_vals,
            materials,
            lorentz_coeffs,
            context.grid,
            context.pec_axes,
            context.pec_faces,
            context.src_meta,
            context.prb_meta,
        )
        if return_trace:
            return (next_state, next_lorentz_state), (probe_out, state, lorentz_state_local)
        return (next_state, next_lorentz_state), probe_out

    (final_state, final_lorentz_state), outputs = jax.lax.scan(
        body,
        (context.initial_state, lorentz_state),
        src_waveforms,
    )
    if return_trace:
        time_series, states_before, lorentz_states_before = outputs
        return final_state, final_lorentz_state, time_series, states_before, lorentz_states_before
    return final_state, final_lorentz_state, outputs


def _run_uniform_lorentz_cpml_time_series(
    context: Phase1HybridContext,
    materials: MaterialArrays,
    lorentz_coeffs: LorentzCoeffs,
    lorentz_state: LorentzState,
    src_waveforms: jnp.ndarray,
    *,
    return_trace: bool = False,
):
    """Run the extracted time-series seam with Lorentz, CPML, and PEC enforcement."""

    assert context.cpml_params is not None

    def body(carry, src_vals):
        state, cpml_state, lorentz_state_local = carry
        next_state, next_cpml_state, next_lorentz_state, probe_out = _uniform_lorentz_cpml_step(
            state,
            cpml_state,
            lorentz_state_local,
            src_vals,
            materials,
            lorentz_coeffs,
            context.grid,
            context.cpml_params,
            context.cpml_axes,
            context.pec_axes,
            context.pec_faces,
            context.src_meta,
            context.prb_meta,
        )
        if return_trace:
            return (next_state, next_cpml_state, next_lorentz_state), (
                probe_out,
                state,
                cpml_state,
                lorentz_state_local,
            )
        return (next_state, next_cpml_state, next_lorentz_state), probe_out

    (final_state, final_cpml_state, final_lorentz_state), outputs = jax.lax.scan(
        body,
        (context.initial_state, _zero_cpml_state(context.grid), lorentz_state),
        src_waveforms,
    )
    if return_trace:
        time_series, states_before, cpml_states_before, lorentz_states_before = outputs
        return (
            final_state,
            final_cpml_state,
            final_lorentz_state,
            time_series,
            states_before,
            cpml_states_before,
            lorentz_states_before,
        )
    return final_state, final_cpml_state, final_lorentz_state, outputs


def _uniform_lossless_pec_step(
    state: Phase1FieldState,
    src_vals: jnp.ndarray,
    coeffs: UpdateCoeffs,
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
    *,
    materials: MaterialArrays | None = None,
    grid: Grid | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    pec_axes: str = "",
    pec_faces: tuple[str, ...] = (),
) -> tuple[Phase1FieldState, jnp.ndarray]:
    """Extracted Phase 1 scan step: fast Yee step + source injection + probes."""

    if periodic != (False, False, False):
        assert materials is not None
        assert grid is not None
        fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=periodic)
        fdtd = update_e(fdtd, materials, grid.dt, grid.dx, periodic=periodic)
        if pec_axes:
            fdtd = apply_pec(fdtd, axes=pec_axes)
        if pec_faces:
            fdtd = apply_pec_faces(fdtd, pec_faces)
    else:
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
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[Phase1FieldState, CPMLState, jnp.ndarray]:
    """Extracted CPML scan step: Yee + CPML + PEC faces + sources + probes."""

    fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=periodic)
    fdtd, next_cpml_state = apply_cpml_h(
        fdtd,
        cpml_params,
        cpml_state,
        grid,
        cpml_axes,
        materials=materials,
    )
    fdtd = update_e(fdtd, materials, grid.dt, grid.dx, periodic=periodic)
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


def _uniform_lorentz_pec_step(
    state: Phase1FieldState,
    lorentz_state: LorentzState,
    src_vals: jnp.ndarray,
    materials: MaterialArrays,
    lorentz_coeffs: LorentzCoeffs,
    grid: Grid,
    pec_axes: str,
    pec_faces: tuple[str, ...],
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> tuple[Phase1FieldState, LorentzState, jnp.ndarray]:
    """Extracted Lorentz scan step: Yee H + Lorentz E + PEC + sources + probes."""

    fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=(False, False, False))
    fdtd, next_lorentz_state = update_e_lorentz(
        fdtd,
        lorentz_coeffs,
        lorentz_state,
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
    return next_state, next_lorentz_state, probe_out


def _uniform_lorentz_cpml_step(
    state: Phase1FieldState,
    cpml_state: CPMLState,
    lorentz_state: LorentzState,
    src_vals: jnp.ndarray,
    materials: MaterialArrays,
    lorentz_coeffs: LorentzCoeffs,
    grid: Grid,
    cpml_params: CPMLAxisParams,
    cpml_axes: str,
    pec_axes: str,
    pec_faces: tuple[str, ...],
    src_meta: tuple[tuple[int, int, int, str], ...],
    prb_meta: tuple[tuple[int, int, int, str], ...],
) -> tuple[Phase1FieldState, CPMLState, LorentzState, jnp.ndarray]:
    """Extracted Lorentz+CPML scan step: Yee H + CPML + Lorentz E + PEC + sources + probes."""

    fdtd = update_h(_to_fdtd(state), materials, grid.dt, grid.dx, periodic=(False, False, False))
    fdtd, next_cpml_state = apply_cpml_h(
        fdtd,
        cpml_params,
        cpml_state,
        grid,
        cpml_axes,
        materials=materials,
    )
    fdtd, next_lorentz_state = update_e_lorentz(
        fdtd,
        lorentz_coeffs,
        lorentz_state,
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
    return next_state, next_cpml_state, next_lorentz_state, probe_out


def _coeffs_from_eps(context: Phase1HybridContext, eps_r: jnp.ndarray) -> UpdateCoeffs:
    materials = _materials_from_eps(context, eps_r)
    return _coeffs_from_materials(context, materials)


def _materials_from_eps(context: Phase1HybridContext, eps_r: jnp.ndarray) -> MaterialArrays:
    return MaterialArrays(
        eps_r=eps_r,
        sigma=context.sigma,
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


def _runtime_lorentz(
    dt: float,
    materials: MaterialArrays,
    lorentz_spec: tuple | None,
) -> tuple[LorentzCoeffs, LorentzState]:
    assert lorentz_spec is not None
    poles, masks = lorentz_spec
    return init_lorentz(poles, materials, dt, mask=masks)


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
        sigma = context.sigma[i, j, k]
        loss = sigma * jnp.float32(context.dt) / (jnp.float32(2.0) * eps)
        cb = (jnp.float32(context.dt) / eps) / (jnp.float32(1.0) + loss)
        if isinstance(context.grid, NonUniformGrid):
            dx_local = context.grid.dx_arr[i]
            dy_local = context.grid.dy_arr[j]
            dz_local = context.grid.dz[k]
            return raw_src_vals * cb / (dx_local * dy_local * dz_local)
        return raw_src_vals * cb

    if raw_src_vals.shape[0] == 0:
        return raw_src_vals

    scaled = []
    for idx, (i, j, k, _) in enumerate(context.src_meta):
        eps = eps_r[i, j, k] * jnp.float32(EPS_0)
        sigma = context.sigma[i, j, k]
        loss = sigma * jnp.float32(context.dt) / (jnp.float32(2.0) * eps)
        cb = (jnp.float32(context.dt) / eps) / (jnp.float32(1.0) + loss)
        if isinstance(context.grid, NonUniformGrid):
            dx_local = context.grid.dx_arr[i]
            dy_local = context.grid.dy_arr[j]
            dz_local = context.grid.dz[k]
            scaled.append(raw_src_vals[idx] * cb / (dx_local * dy_local * dz_local))
        else:
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


def _add_field_states(lhs: Phase1FieldState, rhs: Phase1FieldState) -> Phase1FieldState:
    return Phase1FieldState(*(left + right for left, right in zip(lhs, rhs)))


def _zero_cpml_state(grid: Grid) -> CPMLState:
    _, cpml_state = init_cpml(grid, pec_faces=set(getattr(grid, "pec_faces", set())))
    return cpml_state


def _zero_debye_state(state: DebyeState) -> DebyeState:
    return DebyeState(
        px=jnp.zeros_like(state.px),
        py=jnp.zeros_like(state.py),
        pz=jnp.zeros_like(state.pz),
    )


def _zero_lorentz_state(state: LorentzState) -> LorentzState:
    return LorentzState(
        px=jnp.zeros_like(state.px),
        py=jnp.zeros_like(state.py),
        pz=jnp.zeros_like(state.pz),
        px_prev=jnp.zeros_like(state.px_prev),
        py_prev=jnp.zeros_like(state.py_prev),
        pz_prev=jnp.zeros_like(state.pz_prev),
    )
