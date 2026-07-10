"""Validation helpers for RF port observables.

This module validates the *observable contract* of S-matrix-like port data:
array schema, frequency metadata, finite complex values, and caller-selected
physical invariants such as passivity and reciprocity.  It intentionally does
not run FDTD or calibrate ports; solver/physics validation remains in the
crossval and physics test suites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PortSMatrixObservable:
    """Canonical S-matrix-like RF port observable.

    Parameters
    ----------
    s_params
        Complex array with shape ``(n_ports, n_ports, n_freqs)``.
    freqs
        One-dimensional frequency array in Hz with length ``n_freqs``.
    port_names
        Unique names for the ``n_ports`` rows/columns.
    source
        Human-readable provenance label used in validation diagnostics.
    """

    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    source: str = "port_smatrix"


@dataclass(frozen=True)
class PortDumpMetadata:
    """Metadata for a raw port-observable dump.

    The schema is intentionally explicit because dump replays are physics
    evidence, not just regression fixtures.  At minimum, claims-bearing dumps
    should record the commit, geometry/material/grid/boundary setup, port and
    reference-plane definitions, waveform, time step, frequency grid, raw
    phasor convention, and the production S-matrix being audited.
    """

    commit_hash: str
    geometry: Mapping[str, Any] = field(default_factory=dict)
    materials: Mapping[str, Any] = field(default_factory=dict)
    grid: Mapping[str, Any] = field(default_factory=dict)
    boundaries: Mapping[str, Any] = field(default_factory=dict)
    port_definitions: tuple[Mapping[str, Any], ...] = ()
    reference_planes: Mapping[str, Any] = field(default_factory=dict)
    waveform: Mapping[str, Any] = field(default_factory=dict)
    dt_s: float | None = None
    frequency_grid_hz: tuple[float, ...] = ()
    raw_phasor_type: str = "V/I"
    phase_convention: str = "exp(-j omega t)"
    current_convention: str = "positive_into_dut"
    reference_plane_shift_convention: str = (
        "port coordinate is positive into the DUT; positive offset moves the "
        "raw measurement plane toward the reported reference plane"
    )
    production_smatrix_schema: str = "S[receiver_port, driven_port, frequency_index]"
    notes: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the metadata."""

        return {
            "schema": "rfx.port_vi_dump",
            "schema_version": 1,
            "commit_hash": self.commit_hash,
            "geometry": dict(self.geometry),
            "materials": dict(self.materials),
            "grid": dict(self.grid),
            "boundaries": dict(self.boundaries),
            "port_definitions": [dict(port) for port in self.port_definitions],
            "reference_planes": dict(self.reference_planes),
            "waveform": dict(self.waveform),
            "dt_s": self.dt_s,
            "frequency_grid_hz": list(self.frequency_grid_hz),
            "raw_phasor_type": self.raw_phasor_type,
            "phase_convention": self.phase_convention,
            "current_convention": self.current_convention,
            "reference_plane_shift_convention": self.reference_plane_shift_convention,
            "production_smatrix_schema": self.production_smatrix_schema,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PortVIDump:
    """Raw V/I phasor dump for independent S-matrix replay.

    Arrays use shape ``(n_driven, n_ports, n_freqs)``.  The replay function
    recomputes ``S[receiver, driven, frequency]`` from these raw phasors and
    never calls the production extractor being audited.
    """

    metadata: Mapping[str, Any]
    freqs: np.ndarray
    voltages: np.ndarray
    currents: np.ndarray
    port_impedances: np.ndarray
    port_names: tuple[str, ...] = ()
    driven_port_indices: tuple[int, ...] = ()
    production_smatrix: np.ndarray | None = None
    reference_plane_offsets_m: np.ndarray | None = None
    propagation_constants: np.ndarray | None = None


@dataclass(frozen=True)
class PortReplayComparison:
    """Comparison between independent replay and production S-matrix output."""

    ok: bool
    max_abs_diff: float
    max_allowed: float
    n_ports: int
    n_freqs: int
    atol: float
    rtol: float

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return (
            f"{status} replay-vs-production: max_abs_diff={self.max_abs_diff:.6g}, "
            f"max_allowed={self.max_allowed:.6g}, ports={self.n_ports}, "
            f"freqs={self.n_freqs}, atol={self.atol:g}, rtol={self.rtol:g}"
        )


@dataclass(frozen=True)
class PortValidationIssue:
    """One validation diagnostic for an RF port observable."""

    code: str
    message: str
    severity: str = "error"
    frequency_index: int | None = None
    port_indices: tuple[int, ...] | None = None
    value: float | None = None
    limit: float | None = None


@dataclass(frozen=True)
class PortValidationReport:
    """Validation result returned by :func:`validate_port_smatrix`."""

    source: str
    n_ports: int = 0
    n_freqs: int = 0
    port_names: tuple[str, ...] = ()
    issues: tuple[PortValidationIssue, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether all error-severity checks passed."""

        return not any(issue.severity == "error" for issue in self.issues)

    def by_code(self, code: str) -> tuple[PortValidationIssue, ...]:
        """Return all issues with diagnostic ``code``."""

        return tuple(issue for issue in self.issues if issue.code == code)

    def summary(self) -> str:
        """Compact human-readable validation summary."""

        status = "PASS" if self.ok else "FAIL"
        if not self.issues:
            return (
                f"{status} {self.source}: {self.n_ports} port(s), "
                f"{self.n_freqs} frequency point(s)"
            )
        details = "; ".join(
            f"{issue.code}: {issue.message}" for issue in self.issues[:5]
        )
        if len(self.issues) > 5:
            details += f"; ... ({len(self.issues) - 5} more)"
        return (
            f"{status} {self.source}: {self.n_ports} port(s), "
            f"{self.n_freqs} frequency point(s); {details}"
        )

    def raise_for_failure(self) -> None:
        """Raise ``AssertionError`` when the report contains an error."""

        if not self.ok:
            raise AssertionError(self.summary())


def _coerce_source(obj: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    if obj is None:
        return "port_smatrix"
    return type(obj).__name__


def _extract_s_params(obj: Any, explicit: Any) -> Any:
    if explicit is not None:
        return explicit
    if obj is None:
        raise ValueError("s_params must be provided when no result object is passed")
    if hasattr(obj, "s_params"):
        s_params = getattr(obj, "s_params")
        if s_params is not None:
            return s_params
    if hasattr(obj, "S"):
        s_params = getattr(obj, "S")
        if s_params is not None:
            return s_params
    raise ValueError(
        "could not find S-parameter data; pass s_params=... or an object "
        "with non-None .s_params or .S"
    )


def _extract_freqs(obj: Any, explicit: Any) -> Any:
    if explicit is not None:
        return explicit
    if obj is not None and hasattr(obj, "freqs"):
        freqs = getattr(obj, "freqs")
        if freqs is not None:
            return freqs
    raise ValueError(
        "frequency metadata is required; pass freqs=... or an object with .freqs"
    )


def _default_port_names(n_ports: int) -> tuple[str, ...]:
    return tuple(f"port{i}" for i in range(n_ports))


def _extract_port_names(obj: Any, explicit: Any, n_ports: int) -> tuple[str, ...]:
    raw = explicit
    if raw is None and obj is not None and hasattr(obj, "port_names"):
        raw = getattr(obj, "port_names")
    if raw is None:
        return _default_port_names(n_ports)
    names = tuple(str(name) for name in raw)
    return names


def _canonical_s_params(raw_s: Any, n_freqs: int) -> np.ndarray:
    """Convert common rfx S-output layouts into ``(n, n, nf)``.

    Accepted layouts:
    - ``(nf,)``: one-port reflection vector.
    - ``(n_ports, nf)``: per-port reflection vectors; placed on the diagonal.
    - ``(n_ports, n_ports)`` with ``nf == 1``: single-frequency full matrix.
    - ``(n_ports, n_ports, nf)``: full S-matrix.
    """

    arr = np.asarray(raw_s, dtype=np.complex128)
    if arr.ndim == 1:
        return arr.reshape(1, 1, arr.shape[0])
    if arr.ndim == 2:
        if n_freqs == 1 and arr.shape[0] == arr.shape[1]:
            return arr[:, :, np.newaxis]
        if arr.shape[1] == n_freqs:
            out = np.zeros((arr.shape[0], arr.shape[0], n_freqs), dtype=arr.dtype)
            diag = np.arange(arr.shape[0])
            out[diag, diag, :] = arr
            return out
        raise ValueError(
            "2D S-parameter arrays must be either (n_ports, n_freqs) "
            "reflection vectors or an (n_ports, n_ports) matrix with one frequency"
        )
    if arr.ndim == 3:
        return arr
    raise ValueError(
        f"S-parameter array must be 1D, 2D, or 3D; got shape {arr.shape}"
    )


def normalize_port_smatrix(
    obj: Any | None = None,
    *,
    s_params: Any | None = None,
    freqs: Any | None = None,
    port_names: Any | None = None,
    source: str | None = None,
) -> PortSMatrixObservable:
    """Normalize rfx S-matrix-like outputs into a canonical observable.

    The function uses duck typing so it works for:
    - :class:`rfx.api.Result` / ``Result.s_params``
    - :class:`rfx.api.WaveguideSMatrixResult` / ``.s_params``
    - :class:`rfx.api.MSLSMatrixResult` / ``.S``
    - direct ``s_params=...`` and ``freqs=...`` arrays

    It raises ``ValueError`` for schema problems.  Use
    :func:`validate_port_smatrix` when you want a non-raising report.
    """

    source_label = _coerce_source(obj, source)
    raw_freqs = _extract_freqs(obj, freqs)
    freqs_arr = np.asarray(raw_freqs, dtype=np.float64)
    if freqs_arr.ndim != 1:
        raise ValueError(f"freqs must be 1D; got shape {freqs_arr.shape}")
    raw_s = _extract_s_params(obj, s_params)
    s_arr = _canonical_s_params(raw_s, int(freqs_arr.shape[0]))
    if s_arr.ndim != 3 or s_arr.shape[0] != s_arr.shape[1]:
        raise ValueError(
            "canonical S-parameter data must have shape "
            f"(n_ports, n_ports, n_freqs); got {s_arr.shape}"
        )
    if s_arr.shape[2] != freqs_arr.shape[0]:
        raise ValueError(
            f"frequency length mismatch: S has {s_arr.shape[2]} frequency "
            f"point(s), freqs has {freqs_arr.shape[0]}"
        )
    names = _extract_port_names(obj, port_names, int(s_arr.shape[0]))
    return PortSMatrixObservable(
        s_params=s_arr,
        freqs=freqs_arr,
        port_names=names,
        source=source_label,
    )


def validate_port_smatrix(
    obj: Any | None = None,
    *,
    s_params: Any | None = None,
    freqs: Any | None = None,
    port_names: Any | None = None,
    source: str | None = None,
    check_passivity: bool = True,
    passivity_limit: float = 1.0,
    passivity_tol: float = 1e-9,
    check_reciprocity: bool = False,
    reciprocity_atol: float = 1e-9,
    reciprocity_rtol: float = 1e-6,
    require_positive_freqs: bool = True,
    require_strictly_increasing_freqs: bool = True,
) -> PortValidationReport:
    """Validate schema and selected invariants for an RF port observable.

    Thresholds are caller-controlled.  Defaults are intentionally strict for
    synthetic/unit checks; integration tests may pass looser tolerances that
    match their validated physics envelope.
    """

    source_label = _coerce_source(obj, source)
    try:
        obs = normalize_port_smatrix(
            obj,
            s_params=s_params,
            freqs=freqs,
            port_names=port_names,
            source=source_label,
        )
    except (TypeError, ValueError) as exc:
        return PortValidationReport(
            source=source_label,
            issues=(
                PortValidationIssue(
                    code="invalid_schema",
                    message=str(exc),
                ),
            ),
        )

    s = obs.s_params
    f = obs.freqs
    names = obs.port_names
    n_ports = int(s.shape[0])
    n_freqs = int(s.shape[2])
    issues: list[PortValidationIssue] = []
    metrics: dict[str, float] = {
        "n_ports": float(n_ports),
        "n_freqs": float(n_freqs),
    }

    if len(names) != n_ports:
        issues.append(PortValidationIssue(
            code="port_name_count",
            message=(
                f"expected {n_ports} port name(s), got {len(names)}"
            ),
            value=float(len(names)),
            limit=float(n_ports),
        ))
    elif len(set(names)) != len(names):
        issues.append(PortValidationIssue(
            code="duplicate_port_names",
            message=f"port names must be unique; got {names!r}",
        ))
    if n_ports == 0:
        issues.append(PortValidationIssue(
            code="empty_ports",
            message="S-parameter data must contain at least one port",
        ))

    if not np.all(np.isfinite(f)):
        issues.append(PortValidationIssue(
            code="nonfinite_freqs",
            message="frequency metadata contains NaN or Inf",
        ))
    else:
        if n_freqs == 0:
            issues.append(PortValidationIssue(
                code="empty_freqs",
                message="frequency metadata must contain at least one point",
            ))
        if require_positive_freqs and np.any(f <= 0.0):
            issues.append(PortValidationIssue(
                code="nonpositive_freqs",
                message="all RF frequency points must be positive",
                value=float(np.min(f)) if f.size else None,
                limit=0.0,
            ))
        if require_strictly_increasing_freqs and f.size > 1:
            diffs = np.diff(f)
            if np.any(diffs <= 0.0):
                issues.append(PortValidationIssue(
                    code="nonmonotonic_freqs",
                    message="frequency points must be strictly increasing",
                    value=float(np.min(diffs)),
                    limit=0.0,
                ))

    finite_s = np.isfinite(s)
    if not np.all(finite_s):
        count = int(finite_s.size - np.count_nonzero(finite_s))
        issues.append(PortValidationIssue(
            code="nonfinite_sparams",
            message=f"S-parameter data contains {count} NaN/Inf value(s)",
            value=float(count),
            limit=0.0,
        ))

    # Subsequent physical invariant checks are meaningful only on finite,
    # non-empty data.
    data_is_checkable = bool(np.all(finite_s) and n_ports > 0 and n_freqs > 0)
    if data_is_checkable and check_passivity:
        col_power = np.sum(np.abs(s) ** 2, axis=0)  # (n_ports, n_freqs)
        max_idx = np.unravel_index(int(np.argmax(col_power)), col_power.shape)
        max_power = float(col_power[max_idx])
        limit = float(passivity_limit + passivity_tol)
        metrics["max_column_power"] = max_power
        if max_power > limit:
            port_i, freq_i = int(max_idx[0]), int(max_idx[1])
            issues.append(PortValidationIssue(
                code="passivity_violation",
                message=(
                    f"max column power {max_power:.6g} exceeds "
                    f"limit {limit:.6g} at driven port {port_i}, "
                    f"frequency index {freq_i}"
                ),
                frequency_index=freq_i,
                port_indices=(port_i,),
                value=max_power,
                limit=limit,
            ))

    if data_is_checkable and check_reciprocity and n_ports >= 2:
        diff = np.abs(s - np.swapaxes(s, 0, 1))
        scale = np.maximum(
            1.0,
            np.maximum(np.abs(s), np.abs(np.swapaxes(s, 0, 1))),
        )
        allowed = float(reciprocity_atol) + float(reciprocity_rtol) * scale
        excess = diff - allowed
        max_excess_idx = np.unravel_index(int(np.argmax(excess)), excess.shape)
        max_abs_diff = float(diff[max_excess_idx])
        metrics["max_reciprocity_abs_diff"] = max_abs_diff
        if excess[max_excess_idx] > 0.0:
            i, j, k = (int(max_excess_idx[0]), int(max_excess_idx[1]), int(max_excess_idx[2]))
            issues.append(PortValidationIssue(
                code="reciprocity_violation",
                message=(
                    f"|S[{i},{j}] - S[{j},{i}]| = {max_abs_diff:.6g} "
                    f"exceeds atol+rtol scale at frequency index {k}"
                ),
                frequency_index=k,
                port_indices=(i, j),
                value=max_abs_diff,
                limit=float(allowed[max_excess_idx]),
            ))

    return PortValidationReport(
        source=obs.source,
        n_ports=n_ports,
        n_freqs=n_freqs,
        port_names=names,
        issues=tuple(issues),
        metrics=metrics,
    )


def assert_port_smatrix_valid(*args: Any, **kwargs: Any) -> PortValidationReport:
    """Validate and raise ``AssertionError`` on failure.

    Returns the report on success so tests can inspect metrics without running
    validation twice.
    """

    report = validate_port_smatrix(*args, **kwargs)
    report.raise_for_failure()
    return report


def _as_driven_port_array(name: str, raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.complex128)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n_driven, n_ports, n_freqs) "
            f"or (n_ports, n_freqs); got {arr.shape}"
        )
    return arr


def _impedance_vector(raw: Any, n_ports: int) -> np.ndarray:
    z = np.asarray(raw, dtype=np.complex128)
    if z.ndim == 0:
        z = np.full(n_ports, z.item(), dtype=np.complex128)
    if z.shape != (n_ports,):
        raise ValueError(
            f"port_impedances must be scalar or shape ({n_ports},); got {z.shape}"
        )
    if np.any(z == 0):
        raise ValueError("port_impedances must be nonzero")
    return z


def _metadata_to_mapping(metadata: PortDumpMetadata | Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, PortDumpMetadata):
        return metadata.to_jsonable()
    return dict(metadata)


def replay_smatrix_from_vi_dump(
    voltages: Any,
    currents: Any,
    *,
    freqs: Any | None = None,
    port_impedances: Any = 50.0,
    port_names: Any | None = None,
    driven_port_indices: Any | None = None,
    current_convention: str = "positive_into_dut",
    reference_plane_offsets_m: Any | None = None,
    propagation_constants: Any | None = None,
    source: str = "vi_dump_replay",
) -> PortSMatrixObservable:
    """Replay an S-matrix from raw V/I phasors.

    This is an independent post-processor for E3 evidence.  It uses the
    power-wave split, role-selected per port to mirror the production
    lumped decomposer (issue #308):

    ``a = (V + Z0 I) / (2 sqrt(Z0))`` (incident, at the DRIVEN port),
    ``b = (V - Z0 I) / (2 sqrt(Z0))`` (reflected, at the DRIVEN port), and
    ``b_recv = -(V + Z0 I) / (2 sqrt(Z0))`` (arriving, at a PASSIVE
    receive port — the production receive channel ``(V_fdtd - Z0 I)``
    expressed in this dump's into-DUT voltage convention ``V = -V_fdtd``;
    the overall sign is pinned empirically by the DC falsifier on the
    canonical thru, S21(DC) -> +1)

    with current positive **into** the DUT by default.  If the dump records
    current positive out of the DUT, set ``current_convention="positive_out_of_dut"``
    and the sign is flipped before wave decomposition.

    Optional reference-plane shifts are explicit metadata, not hidden magic:
    ``reference_plane_offsets_m[i] > 0`` means the reported reference plane
    is farther into the DUT than the raw measurement plane in port ``i``'s
    local coordinate.  With propagation constant ``gamma`` this applies
    ``a_ref = a_raw exp(-gamma d)`` and
    ``b_ref = b_raw exp(+gamma d)``.
    """

    v = _as_driven_port_array("voltages", voltages)
    i = _as_driven_port_array("currents", currents)
    if v.shape != i.shape:
        raise ValueError(f"voltages and currents shape mismatch: {v.shape} vs {i.shape}")

    n_driven, n_ports, n_freqs = v.shape
    if freqs is None:
        freqs_arr = np.arange(n_freqs, dtype=np.float64)
    else:
        freqs_arr = np.asarray(freqs, dtype=np.float64)
        if freqs_arr.shape != (n_freqs,):
            raise ValueError(
                f"freqs must have shape ({n_freqs},); got {freqs_arr.shape}"
            )

    convention = current_convention.lower().replace("-", "_")
    if convention in {"positive_out_of_dut", "out_of_dut", "out"}:
        i = -i
    elif convention not in {"positive_into_dut", "into_dut", "in"}:
        raise ValueError(
            "current_convention must be 'positive_into_dut' or "
            "'positive_out_of_dut'"
        )

    z = _impedance_vector(port_impedances, n_ports)
    z_view = z.reshape(1, n_ports, 1)
    sqrt_z = np.sqrt(z_view)
    a = (v + z_view * i) / (2.0 * sqrt_z)
    b = (v - z_view * i) / (2.0 * sqrt_z)
    # Passive-receive channel (issue #308): the production receive b-wave in
    # this dump's into-DUT convention.  Selected per role in the loop below.
    b_recv = -(v + z_view * i) / (2.0 * sqrt_z)

    if reference_plane_offsets_m is not None:
        if propagation_constants is None:
            raise ValueError(
                "propagation_constants are required when reference-plane "
                "offsets are provided"
            )
        offsets = np.asarray(reference_plane_offsets_m, dtype=np.float64)
        if offsets.shape != (n_ports,):
            raise ValueError(
                f"reference_plane_offsets_m must have shape ({n_ports},); "
                f"got {offsets.shape}"
            )
        gamma = np.asarray(propagation_constants, dtype=np.complex128)
        if gamma.ndim == 1:
            gamma = np.broadcast_to(gamma.reshape(1, n_freqs), (n_ports, n_freqs))
        if gamma.shape != (n_ports, n_freqs):
            raise ValueError(
                "propagation_constants must have shape "
                f"({n_ports}, {n_freqs}) or ({n_freqs},); got {gamma.shape}"
            )
        shift = np.exp(gamma.reshape(1, n_ports, n_freqs) * offsets.reshape(1, n_ports, 1))
        a = a / shift
        b = b * shift
        b_recv = b_recv * shift

    if driven_port_indices is None:
        driven = tuple(range(n_driven))
    else:
        driven = tuple(int(idx) for idx in driven_port_indices)
    if len(driven) != n_driven:
        raise ValueError(
            f"driven_port_indices length {len(driven)} does not match "
            f"n_driven {n_driven}"
        )
    if any(idx < 0 or idx >= n_ports for idx in driven):
        raise ValueError(f"driven_port_indices out of range for {n_ports} ports: {driven}")

    s = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    for drive_i, driven_port in enumerate(driven):
        denom = a[drive_i, driven_port, :]
        # Role-selected numerator (issue #308): the reflected-wave channel at
        # the driven port, the passive-receive channel everywhere else.
        numer = b_recv[drive_i, :, :].copy()
        numer[driven_port, :] = b[drive_i, driven_port, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            s[:, driven_port, :] = np.divide(
                numer,
                denom.reshape(1, n_freqs),
                out=np.full((n_ports, n_freqs), np.nan + 1j * np.nan),
                where=np.abs(denom.reshape(1, n_freqs)) > 0.0,
            )

    names = tuple(str(name) for name in port_names) if port_names is not None else _default_port_names(n_ports)
    return PortSMatrixObservable(
        s_params=s,
        freqs=freqs_arr,
        port_names=names,
        source=source,
    )


def compare_replayed_smatrix(
    replayed: PortSMatrixObservable | Any,
    production: Any,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> PortReplayComparison:
    """Compare an independent replay against production S-matrix output."""

    replay_obs = (
        replayed
        if isinstance(replayed, PortSMatrixObservable)
        else normalize_port_smatrix(replayed, source="replayed")
    )
    prod_obs = normalize_port_smatrix(production, source="production")
    if replay_obs.s_params.shape != prod_obs.s_params.shape:
        raise ValueError(
            "replayed and production S matrices have different shapes: "
            f"{replay_obs.s_params.shape} vs {prod_obs.s_params.shape}"
        )
    if replay_obs.freqs.shape != prod_obs.freqs.shape or not np.allclose(
        replay_obs.freqs, prod_obs.freqs, rtol=0.0, atol=0.0
    ):
        raise ValueError("replayed and production frequency grids differ")

    diff = np.abs(replay_obs.s_params - prod_obs.s_params)
    scale = np.maximum(np.abs(replay_obs.s_params), np.abs(prod_obs.s_params))
    allowed = float(atol) + float(rtol) * scale
    return PortReplayComparison(
        ok=bool(np.all(diff <= allowed)),
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        max_allowed=float(np.max(allowed)) if allowed.size else float(atol),
        n_ports=int(replay_obs.s_params.shape[0]),
        n_freqs=int(replay_obs.s_params.shape[2]),
        atol=float(atol),
        rtol=float(rtol),
    )


def save_port_vi_dump_npz(
    path: str | Path,
    *,
    voltages: Any,
    currents: Any,
    freqs: Any,
    port_impedances: Any = 50.0,
    metadata: PortDumpMetadata | Mapping[str, Any] | None = None,
    port_names: Any | None = None,
    driven_port_indices: Any | None = None,
    production_smatrix: Any | None = None,
    reference_plane_offsets_m: Any | None = None,
    propagation_constants: Any | None = None,
) -> None:
    """Write a compact ``.npz`` V/I dump for independent replay."""

    v = _as_driven_port_array("voltages", voltages)
    i = _as_driven_port_array("currents", currents)
    if v.shape != i.shape:
        raise ValueError(f"voltages and currents shape mismatch: {v.shape} vs {i.shape}")
    n_driven, n_ports, n_freqs = v.shape
    f = np.asarray(freqs, dtype=np.float64)
    if f.shape != (n_freqs,):
        raise ValueError(f"freqs must have shape ({n_freqs},); got {f.shape}")
    z = _impedance_vector(port_impedances, n_ports)
    names = np.asarray(
        tuple(str(name) for name in port_names)
        if port_names is not None else _default_port_names(n_ports),
        dtype=object,
    )
    driven = np.asarray(
        tuple(range(n_driven)) if driven_port_indices is None else tuple(driven_port_indices),
        dtype=np.int64,
    )
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(_metadata_to_mapping(metadata))),
        "freqs_hz": f,
        "voltages": v,
        "currents": i,
        "port_impedances_ohm": z,
        "port_names": names,
        "driven_port_indices": driven,
    }
    if production_smatrix is not None:
        payload["production_smatrix"] = np.asarray(production_smatrix, dtype=np.complex128)
    if reference_plane_offsets_m is not None:
        payload["reference_plane_offsets_m"] = np.asarray(reference_plane_offsets_m, dtype=np.float64)
    if propagation_constants is not None:
        payload["propagation_constants"] = np.asarray(propagation_constants, dtype=np.complex128)
    np.savez(path, **payload)


def load_port_vi_dump_npz(path: str | Path) -> PortVIDump:
    """Load a compact ``.npz`` V/I dump."""

    with np.load(path, allow_pickle=True) as data:
        metadata = (
            json.loads(str(data["metadata_json"].item()))
            if "metadata_json" in data
            else {}
        )
        production = data["production_smatrix"] if "production_smatrix" in data else None
        offsets = data["reference_plane_offsets_m"] if "reference_plane_offsets_m" in data else None
        gamma = data["propagation_constants"] if "propagation_constants" in data else None
        return PortVIDump(
            metadata=metadata,
            freqs=np.asarray(data["freqs_hz"], dtype=np.float64),
            voltages=np.asarray(data["voltages"], dtype=np.complex128),
            currents=np.asarray(data["currents"], dtype=np.complex128),
            port_impedances=np.asarray(data["port_impedances_ohm"], dtype=np.complex128),
            port_names=tuple(str(name) for name in data["port_names"].tolist()),
            driven_port_indices=tuple(int(idx) for idx in data["driven_port_indices"].tolist()),
            production_smatrix=None if production is None else np.asarray(production, dtype=np.complex128),
            reference_plane_offsets_m=None if offsets is None else np.asarray(offsets, dtype=np.float64),
            propagation_constants=None if gamma is None else np.asarray(gamma, dtype=np.complex128),
        )


def replay_smatrix_from_port_vi_dump(dump: PortVIDump) -> PortSMatrixObservable:
    """Replay an S-matrix from a loaded :class:`PortVIDump`."""

    current_convention = str(dump.metadata.get("current_convention", "positive_into_dut"))
    return replay_smatrix_from_vi_dump(
        dump.voltages,
        dump.currents,
        freqs=dump.freqs,
        port_impedances=dump.port_impedances,
        port_names=dump.port_names,
        driven_port_indices=dump.driven_port_indices,
        current_convention=current_convention,
        reference_plane_offsets_m=dump.reference_plane_offsets_m,
        propagation_constants=dump.propagation_constants,
        source="port_vi_dump",
    )
