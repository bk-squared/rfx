"""Phase XIV Strategy B RF-observable validation harness.

This harness adds a source-labelled RF evidence layer after Phase XIII
production promotion.  It deliberately does **not** claim native Strategy B
S-parameters: the current Phase 1 hybrid seam returns ``s_params=None`` and
``freqs=None``.  Required Phase XIV evidence therefore comes from Strategy B
``time_series`` observables and explicit analytic/reference-rfx comparisons,
while Meep/openEMS correlation remains opt-in and skip-aware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from rfx.grid import Grid  # noqa: E402
from rfx.hybrid_adjoint import phase1_forward_result  # noqa: E402
from scripts import phase7_strategy_b_readiness as phase7  # noqa: E402

SCHEMA_VERSION = "phase14.rf_observable_validation.v1"
BENCHMARK_CONTRACT = "phase_xiv_strategy_b_rf_observable_validation"
DEFAULT_PHASE13_BASELINE = Path(".omx/artifacts/phase13_all_promotion.json")
DEFAULT_OUTPUT = Path(".omx/artifacts/phase14_rf_observable_validation.json")
DEFAULT_N_STEPS = 256
DEFAULT_CHECKPOINT_EVERY = 32
DEFAULT_RESONANCE_TOLERANCE = 0.02
DEFAULT_PASSIVITY_TOLERANCE = 1.05
DEFAULT_INPUT_IMPEDANCE_REL_TOLERANCE = 0.1
ALIGNED_CAVITY_A_M = 0.1
ALIGNED_CAVITY_B_M = 0.1
ALIGNED_CAVITY_D_M = 0.05
ALIGNED_CAVITY_DX_M = 0.001
ALIGNED_CAVITY_FREQ_HZ = 299_792_458.0 / 2.0 * math.sqrt(
    (1.0 / ALIGNED_CAVITY_A_M) ** 2 + (1.0 / ALIGNED_CAVITY_B_M) ** 2
)
OPTIONAL_SOLVERS = ("meep", "openems")
CANONICAL_STRATEGY_B_SEAM_FIELDS = (
    "native_s_params_supported",
    "phase1_forward_result_s_params",
    "phase1_forward_result_freqs",
)
FORBIDDEN_NATIVE_SPARAM_KEYS = {
    "native_strategy_b_sparams",
    "native_sparams_supported",
    "strategy_b_native_sparams",
    "strategy_b_native_s_params_supported",
    "strategy_b_native_s_params_present",
}
ALLOWED_REQUIRED_SOURCES = (
    "strategy_b_time_series",
    "strategy_b_ntff_metadata",
    "analytic_reference",
    "standard_rfx_time_series_reference",
    "standard_rfx_sparams_reference",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort outside git.
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json_hash(payload: Any) -> str:
    return _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    )


def worktree_signature() -> dict[str, Any]:
    head = _run_git(["rev-parse", "HEAD"]) or "unknown"
    status = _run_git(["status", "--short"]) or ""
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": _sha256_text(status),
    }


def environment_summary() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "numpy": np.__version__,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase13_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return summary
    return payload


def phase13_baseline_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        record.update(
            {
                "sha256": None,
                "all_eligible_promoted": False,
                "blocked_families": None,
                "promoted_families": None,
                "valid": False,
                "failure_reason": "missing_phase13_baseline",
            }
        )
        return record

    payload = read_json(path)
    summary = _phase13_summary(payload)
    blocked = summary.get("blocked_families")
    promoted = summary.get("promoted_families")
    all_promoted = summary.get("all_eligible_promoted") is True
    no_blocked = isinstance(blocked, list) and len(blocked) == 0
    record.update(
        {
            "sha256": file_sha256(path),
            "all_eligible_promoted": all_promoted,
            "blocked_families": blocked,
            "promoted_families": promoted,
            "valid": bool(all_promoted and no_blocked),
            "failure_reason": None
            if all_promoted and no_blocked
            else "phase13_not_all_eligible_promoted",
        }
    )
    return record


def strategy_b_seam_record() -> dict[str, Any]:
    grid = Grid(freq_max=1e9, domain=(0.003, 0.003, 0.003), dx=0.001, cpml_layers=0)
    result = phase1_forward_result(grid, jnp.zeros((4, 1), dtype=jnp.float32))
    return {
        "native_s_params_supported": result.s_params is not None
        and result.freqs is not None,
        "phase1_forward_result_s_params": result.s_params,
        "phase1_forward_result_freqs": result.freqs,
        "allowed_required_sources": list(ALLOWED_REQUIRED_SOURCES),
        "runtime_surface": "rfx.hybrid_adjoint.phase1_forward_result",
    }


def extract_resonance_frequency(
    time_series: Any,
    dt_s: float,
    *,
    min_samples: int = 16,
    method: str = "fft_peak",
    f_lo_hz: float | None = None,
    f_hi_hz: float | None = None,
    zero_pad_factor: int = 1,
) -> dict[str, Any]:
    """Extract a finite RF spectral peak from a time series.

    The function is intentionally conservative: invalid, constant, or
    under-sampled arrays raise ``ValueError`` rather than returning a bogus peak.
    """

    series = np.asarray(time_series, dtype=np.float64).reshape(-1)
    if series.size < min_samples:
        raise ValueError("insufficient_time_series_samples")
    if not np.isfinite(series).all() or not math.isfinite(float(dt_s)) or dt_s <= 0:
        raise ValueError("nonfinite_time_series_or_dt")

    centered = series - float(np.mean(series))
    peak_abs = float(np.max(np.abs(centered))) if centered.size else 0.0
    energy = float(np.sum(centered**2))
    if peak_abs <= 1e-18 or energy <= 1e-30:
        raise ValueError("no_resolvable_peak")

    # Keep the spectral estimator aligned with the repository's solver
    # cross-validation fixtures (tests.test_meep_crossval.fft_peak_freq): raw
    # time-series FFT, optional zero-padding, and 3-point parabolic peak
    # interpolation.  Hann-windowing suppresses the TM110 bin in the short
    # Strategy B cavity trace and can select a lower transient mode instead.
    n_fft = max(series.size, series.size * max(1, int(zero_pad_factor)))
    spectrum = np.abs(np.fft.rfft(series, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt_s)
    if spectrum.size <= 1:
        raise ValueError("no_positive_frequency_bins")
    search = spectrum.copy()
    search[0] = 0.0
    if f_lo_hz is not None:
        search = np.where(freqs >= float(f_lo_hz), search, 0.0)
    if f_hi_hz is not None:
        search = np.where(freqs <= float(f_hi_hz), search, 0.0)
    peak_index = int(np.argmax(search))
    peak_magnitude = float(spectrum[peak_index])
    if peak_index <= 0 or peak_magnitude <= max(float(np.max(spectrum)) * 1e-12, 1e-30):
        raise ValueError("no_resolvable_peak")
    peak_frequency = float(freqs[peak_index])
    if 0 < peak_index < len(spectrum) - 1:
        alpha = float(spectrum[peak_index - 1])
        beta = float(spectrum[peak_index])
        gamma = float(spectrum[peak_index + 1])
        denom = alpha - 2.0 * beta + gamma
        if beta >= alpha and beta >= gamma and abs(denom) > 1e-30:
            offset = 0.5 * (alpha - gamma) / denom
            if abs(offset) <= 1.0:
                peak_frequency += offset * float(freqs[1] - freqs[0])

    return {
        "method": method,
        "frequency_hz": peak_frequency,
        "peak_bin": peak_index,
        "sample_count": int(series.size),
        "dt_s": float(dt_s),
        "spectral_resolution_hz": float(freqs[1] - freqs[0]),
        "search_band_hz": [f_lo_hz, f_hi_hz],
        "zero_pad_factor": int(zero_pad_factor),
        "peak_magnitude": peak_magnitude,
        "peak_abs_time_domain": peak_abs,
        "energy": energy,
        "finite": True,
    }


def relative_error(measured: float, reference: float) -> float:
    if not math.isfinite(measured) or not math.isfinite(reference):
        return math.inf
    return abs(measured - reference) / max(abs(reference), 1.0)


def passivity_check(
    s11: Any,
    s21: Any | None = None,
    *,
    tolerance: float = DEFAULT_PASSIVITY_TOLERANCE,
) -> dict[str, Any]:
    s11_arr = np.asarray(s11)
    arrays = [s11_arr]
    if s21 is not None:
        arrays.append(np.asarray(s21))
    finite = all(np.isfinite(arr).all() for arr in arrays)
    max_abs_s11 = float(np.max(np.abs(s11_arr))) if s11_arr.size else math.inf
    max_abs_s21 = None
    if s21 is not None:
        s21_arr = np.asarray(s21)
        max_abs_s21 = float(np.max(np.abs(s21_arr))) if s21_arr.size else math.inf
    max_abs = max(max_abs_s11, max_abs_s21 or 0.0)
    passed = bool(finite and max_abs <= tolerance)
    return {
        "observable_source": "standard_rfx_sparams_reference",
        "max_abs_s11": max_abs_s11,
        "max_abs_s21": max_abs_s21,
        "passivity_tolerance": tolerance,
        "finite": finite,
        "passed": passed,
        "failure_reason": None if passed else "passivity_threshold_exceeded",
    }


def bounded_time_series_proxy(
    time_series: Any,
    *,
    excitation_series: Any | None = None,
    tolerance: float = DEFAULT_PASSIVITY_TOLERANCE,
) -> dict[str, Any]:
    series = np.asarray(time_series, dtype=np.float64).reshape(-1)
    finite = bool(series.size and np.isfinite(series).all())
    peak = float(np.max(np.abs(series))) if series.size else math.inf
    total = float(np.sum(series**2)) if series.size else math.inf
    source_peak = None
    bounded_ratio = math.inf
    method = "bounded_normalized_late_peak"
    if excitation_series is not None:
        source = np.asarray(excitation_series, dtype=np.float64).reshape(-1)
        if source.size and np.isfinite(source).all():
            source_peak = float(np.max(np.abs(source)))
            bounded_ratio = peak / max(source_peak, 1e-30)
            method = "bounded_probe_to_source_peak"
    if source_peak is None:
        first_half = series[: max(1, series.size // 2)]
        second_half = series[series.size // 2 :]
        early_peak = float(np.max(np.abs(first_half))) if first_half.size else math.inf
        late_peak = float(np.max(np.abs(second_half))) if second_half.size else math.inf
        bounded_ratio = late_peak / max(early_peak, 1e-30)
    passed = bool(finite and peak > 0.0 and bounded_ratio <= tolerance)
    return {
        "observable_source": "strategy_b_time_series",
        "method": method,
        "peak_abs": peak,
        "source_peak_abs": source_peak,
        "energy": total,
        "bounded_ratio": float(bounded_ratio),
        "passivity_tolerance": tolerance,
        "finite": finite,
        "passed": passed,
        "failure_reason": None if passed else "bounded_proxy_threshold_exceeded",
    }


def input_impedance_check(
    zin_ohm: complex | float | None,
    *,
    z0_ohm: float = 50.0,
    tolerance_relative: float = DEFAULT_INPUT_IMPEDANCE_REL_TOLERANCE,
    applicable: bool = True,
    not_applicable_reason: str | None = None,
) -> dict[str, Any]:
    if not applicable:
        reason_ok = bool(not_applicable_reason and not_applicable_reason.strip())
        return {
            "observable_source": "derived_proxy_or_standard_reference",
            "z0_ohm": z0_ohm,
            "zin_ohm": None,
            "relative_error": None,
            "tolerance_relative": tolerance_relative,
            "applicable": False,
            "not_applicable_reason": not_applicable_reason,
            "passed": reason_ok,
            "failure_reason": None if reason_ok else "missing_not_applicable_reason",
        }

    if zin_ohm is None:
        return {
            "observable_source": "derived_proxy_or_standard_reference",
            "z0_ohm": z0_ohm,
            "zin_ohm": None,
            "relative_error": math.inf,
            "tolerance_relative": tolerance_relative,
            "applicable": True,
            "passed": False,
            "failure_reason": "missing_input_impedance",
        }

    zin = complex(zin_ohm)
    finite = math.isfinite(zin.real) and math.isfinite(zin.imag)
    error = abs(zin - z0_ohm) / max(abs(z0_ohm), 1.0) if finite else math.inf
    passed = bool(finite and error <= tolerance_relative)
    return {
        "observable_source": "derived_proxy_or_standard_reference",
        "z0_ohm": z0_ohm,
        "zin_ohm": [zin.real, zin.imag],
        "relative_error": float(error),
        "tolerance_relative": tolerance_relative,
        "applicable": True,
        "passed": passed,
        "failure_reason": None if passed else "input_impedance_tolerance_exceeded",
    }


def _solver_module_name(solver: str) -> str:
    normalized = solver.lower()
    if normalized == "openems":
        return "openEMS"
    if normalized == "meep":
        return "meep"
    raise ValueError(f"unsupported solver {solver!r}")


def _solver_available(solver: str) -> bool:
    return importlib.util.find_spec(_solver_module_name(solver)) is not None


def _subprocess_solver_available(
    solver: str,
    *,
    solver_python: str,
    solver_pythonpath: str | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Return whether the requested solver imports in an isolated Python runtime."""

    if solver.lower() != "meep":
        return _solver_available(solver)
    env = os.environ.copy()
    if solver_pythonpath:
        env["PYTHONPATH"] = solver_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            [solver_python, "-c", "import meep"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except Exception:  # noqa: BLE001 - availability is best-effort evidence.
        return False
    return completed.returncode == 0


def _standalone_meep_cavity_code() -> str:
    return r"""
import json
import math

import meep as mp

C0 = 299_792_458.0
CAVITY_A = 0.1
CAVITY_B = 0.1
CAVITY_D = 0.05
F_ANALYTICAL = (C0 / 2.0) * math.sqrt((1.0 / CAVITY_A) ** 2 + (1.0 / CAVITY_B) ** 2)
unit = 0.01
Lx = CAVITY_A / unit
Ly = CAVITY_B / unit
Lz = CAVITY_D / unit
fcen = F_ANALYTICAL * unit / C0
df = fcen * 0.8
sim = mp.Simulation(
    cell_size=mp.Vector3(Lx, Ly, Lz),
    resolution=10,
    boundary_layers=[],
    default_material=mp.Medium(epsilon=1),
)
sim.sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(Lx / 3.0 - Lx / 2.0, Ly / 3.0 - Ly / 2.0, 0.0),
    )
]
probe_pt = mp.Vector3(2.0 * Lx / 3.0 - Lx / 2.0, 2.0 * Ly / 3.0 - Ly / 2.0, 0.0)
harminv = mp.Harminv(mp.Ez, probe_pt, fcen, df)
sim.run(mp.after_sources(harminv), until_after_sources=200 / fcen)
if not harminv.modes:
    raise SystemExit("Meep Harminv found no modes")
best = max(harminv.modes, key=lambda mode: abs(mode.amp))
print(json.dumps({"frequency_hz": float(best.freq * C0 / unit), "q": float(best.Q)}))
"""


def _run_subprocess_solver(
    solver: str,
    *,
    solver_python: str,
    solver_pythonpath: str | None = None,
    timeout_s: float | None = None,
) -> float:
    if solver.lower() != "meep":
        raise ValueError("subprocess solver runtime is currently implemented for Meep")
    env = os.environ.copy()
    if solver_pythonpath:
        env["PYTHONPATH"] = solver_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [solver_python, "-c", _standalone_meep_cavity_code()],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "subprocess solver failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return float(payload["frequency_hz"])
    raise RuntimeError("subprocess solver did not emit a frequency JSON line")


def _default_solver_runner(
    solver: str,
    *,
    solver_python: str | None = None,
    solver_pythonpath: str | None = None,
    solver_timeout_s: float | None = None,
) -> Callable[[Mapping[str, Any]], float]:
    normalized = solver.lower()
    if solver_python is not None and normalized == "meep":
        return lambda _fixture: _run_subprocess_solver(
            normalized,
            solver_python=solver_python,
            solver_pythonpath=solver_pythonpath,
            timeout_s=solver_timeout_s,
        )
    if normalized == "meep":
        from tests.test_meep_crossval import run_meep_cavity

        return lambda _fixture: float(run_meep_cavity())
    if normalized == "openems":
        from tests.test_openems_crossval import run_openems_cavity

        return lambda _fixture: float(run_openems_cavity())
    raise ValueError(f"unsupported solver {solver!r}")


def skipped_solver_record(
    solver: str,
    *,
    required: bool,
    reason: str,
    strategy_b_frequency_hz: float | None = None,
    tolerance: float = DEFAULT_RESONANCE_TOLERANCE,
    fixture_id: str | None = None,
) -> dict[str, Any]:
    return {
        "solver": solver,
        "available": False,
        "version": None,
        "executed": False,
        "required": required,
        "skipped_reason": reason,
        "observable": "resonance_frequency",
        "fixture_id": fixture_id,
        "reference_frequency_hz": None,
        "strategy_b_frequency_hz": strategy_b_frequency_hz,
        "relative_error": None,
        "tolerance": tolerance,
        "passed": False if required else None,
        "failure_reason": f"required_solver_{reason}" if required else None,
    }


def _run_optional_solver_correlation(
    solver: str,
    required: bool,
    fixture: Mapping[str, Any],
    *,
    runner: Callable[[Mapping[str, Any]], float] | None = None,
    availability_checker: Callable[[str], bool] = _solver_available,
    tolerance: float = DEFAULT_RESONANCE_TOLERANCE,
    solver_python: str | None = None,
    solver_pythonpath: str | None = None,
    solver_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run or skip optional solver correlation with fail-closed semantics."""

    strategy_b_frequency = fixture.get("strategy_b_frequency_hz")
    fixture_id = fixture.get("fixture_id")
    if (
        availability_checker is _solver_available
        and solver_python is not None
        and solver.lower() == "meep"
    ):
        available = _subprocess_solver_available(
            solver,
            solver_python=solver_python,
            solver_pythonpath=solver_pythonpath,
            timeout_s=solver_timeout_s,
        )
    else:
        available = availability_checker(solver)
    if not available:
        return skipped_solver_record(
            solver,
            required=required,
            reason="not_installed",
            strategy_b_frequency_hz=strategy_b_frequency,
            tolerance=tolerance,
            fixture_id=str(fixture_id) if fixture_id is not None else None,
        )

    runner = runner or _default_solver_runner(
        solver,
        solver_python=solver_python,
        solver_pythonpath=solver_pythonpath,
        solver_timeout_s=solver_timeout_s,
    )
    started = time.perf_counter()
    cwd_before = Path.cwd()
    try:
        reference_frequency = float(runner(fixture))
    except Exception as exc:  # noqa: BLE001 - convert optional solver failures to evidence.
        return {
            "solver": solver,
            "available": True,
            "version": None,
            "executed": True,
            "required": required,
            "skipped_reason": None,
            "observable": "resonance_frequency",
            "fixture_id": fixture_id,
            "reference_frequency_hz": None,
            "strategy_b_frequency_hz": strategy_b_frequency,
            "relative_error": None,
            "tolerance": tolerance,
            "passed": False,
            "failure_reason": "solver_execution_failed",
            "error": str(exc),
            "runtime_s": round(time.perf_counter() - started, 6),
        }
    finally:
        try:
            os.chdir(cwd_before)
        except FileNotFoundError:
            os.chdir(ROOT)

    if strategy_b_frequency is None:
        err = math.inf
    else:
        err = relative_error(float(strategy_b_frequency), reference_frequency)
    passed = bool(math.isfinite(err) and err <= tolerance)
    return {
        "solver": solver,
        "available": True,
        "version": None,
        "executed": True,
        "required": required,
        "skipped_reason": None,
        "observable": "resonance_frequency",
        "fixture_id": fixture_id,
        "reference_frequency_hz": reference_frequency,
        "strategy_b_frequency_hz": strategy_b_frequency,
        "relative_error": err,
        "tolerance": tolerance,
        "passed": passed,
        "failure_reason": None if passed else "solver_correlation_tolerance_exceeded",
        "runtime_s": round(time.perf_counter() - started, 6),
    }


def _contains_forbidden_native_sparam_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_NATIVE_SPARAM_KEYS:
                return True
            if key == "s_params" and child is not None:
                return True
            if _contains_forbidden_native_sparam_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_native_sparam_key(item) for item in value)
    return False


def _source_label_valid(observable: Mapping[str, Any]) -> bool:
    source = observable.get("observable_source")
    if not isinstance(source, str) or not source:
        return False
    return source in ALLOWED_REQUIRED_SOURCES or source == "derived_proxy_or_standard_reference"


def evaluate_artifact_gates(artifact: Mapping[str, Any]) -> dict[str, Any]:
    failed: list[str] = []
    warnings: list[str] = []

    baseline = artifact.get("phase13_baseline", {})
    if not isinstance(baseline, Mapping) or baseline.get("valid") is not True:
        failed.append("phase13_baseline_not_green")

    seam = artifact.get("strategy_b_seam", {})
    if not isinstance(seam, Mapping):
        failed.append("strategy_b_seam_missing")
    else:
        missing = [
            field for field in CANONICAL_STRATEGY_B_SEAM_FIELDS if field not in seam
        ]
        if missing:
            failed.append("strategy_b_seam_canonical_fields_missing")
        if seam.get("native_s_params_supported") is not False:
            failed.append("native_strategy_b_sparams_claimed_without_support")
        if seam.get("phase1_forward_result_s_params") is not None:
            failed.append("phase1_forward_result_s_params_not_null")
        if seam.get("phase1_forward_result_freqs") is not None:
            failed.append("phase1_forward_result_freqs_not_null")

    if _contains_forbidden_native_sparam_key(artifact):
        failed.append("forbidden_native_sparam_artifact_key")

    fixtures = artifact.get("fixtures", [])
    required_fixtures = [fx for fx in fixtures if isinstance(fx, Mapping) and fx.get("required")]
    if not required_fixtures:
        failed.append("required_fixture_missing")

    resonance_ok = False
    time_series_ok = False
    passivity_ok = False
    input_impedance_ok = False
    source_labels_ok = True
    for fixture in required_fixtures:
        observables = fixture.get("observables", {})
        if not isinstance(observables, Mapping):
            failed.append("fixture_observables_missing")
            continue
        for observable in observables.values():
            if isinstance(observable, Mapping) and not _source_label_valid(observable):
                source_labels_ok = False
        resonance = observables.get("resonance", {})
        if isinstance(resonance, Mapping):
            time_series_ok = time_series_ok or (
                resonance.get("observable_source") == "strategy_b_time_series"
                and resonance.get("finite") is True
                and math.isfinite(float(resonance.get("frequency_hz", math.inf)))
            )
            resonance_ok = resonance_ok or (
                resonance.get("passed") is True
                and resonance.get("relative_error", math.inf)
                <= resonance.get("tolerance", 0.0)
            )
        bounded = observables.get("bounded_reflection_proxy", {})
        sparams_ref = observables.get("sparams_reference", {})
        if isinstance(bounded, Mapping) and bounded.get("passed") is True:
            passivity_ok = True
        if isinstance(sparams_ref, Mapping) and sparams_ref.get("passed") is True:
            passivity_ok = True
        impedance = observables.get("input_impedance", {})
        if isinstance(impedance, Mapping):
            if impedance.get("applicable") is False:
                input_impedance_ok = input_impedance_ok or bool(
                    impedance.get("not_applicable_reason")
                )
            else:
                input_impedance_ok = input_impedance_ok or impedance.get("passed") is True

    if not source_labels_ok:
        failed.append("observable_source_missing_or_ambiguous")
    if not time_series_ok:
        failed.append("strategy_b_time_series_evidence_missing")
    if not resonance_ok:
        failed.append("resonance_reference_correlation_failed")
    if not passivity_ok:
        failed.append("passivity_or_bounded_reflection_failed")
    if not input_impedance_ok:
        failed.append("input_impedance_reference_missing_or_invalid")

    optional_records = artifact.get("optional_solver_correlation", [])
    if not isinstance(optional_records, list):
        failed.append("optional_solver_records_not_list")
        optional_records = []
    if not optional_records:
        failed.append("optional_solver_status_missing")
    for record in optional_records:
        if not isinstance(record, Mapping):
            failed.append("optional_solver_record_invalid")
            continue
        if record.get("required") is True and record.get("passed") is not True:
            failed.append("optional_solver_required_but_not_passed")
        if record.get("executed") is False and not record.get("skipped_reason"):
            failed.append("optional_solver_skip_reason_missing")

    gates = {
        "phase13_baseline_valid": "phase13_baseline_not_green" not in failed,
        "strategy_b_time_series_rf_observables_valid": time_series_ok,
        "resonance_reference_correlation_valid": resonance_ok,
        "passivity_or_bounded_reflection_valid": passivity_ok,
        "input_impedance_reference_valid_or_not_applicable": input_impedance_ok,
        "native_sparams_truthfulness_valid": not any(
            gate
            in {
                "strategy_b_seam_missing",
                "strategy_b_seam_canonical_fields_missing",
                "native_strategy_b_sparams_claimed_without_support",
                "phase1_forward_result_s_params_not_null",
                "phase1_forward_result_freqs_not_null",
                "forbidden_native_sparam_artifact_key",
            }
            for gate in failed
        ),
        "optional_solver_correlation_status_recorded": not any(
            gate
            in {
                "optional_solver_records_not_list",
                "optional_solver_status_missing",
                "optional_solver_record_invalid",
                "optional_solver_skip_reason_missing",
            }
            for gate in failed
        ),
    }
    return {
        "gates": gates,
        "failed_gates": sorted(set(failed)),
        "warnings": warnings,
        "overall_status": "rf_observable_validated_limited"
        if not failed
        else "rf_observable_blocked",
    }


def _run_default_strategy_b_fixture(
    *,
    n_steps: int = DEFAULT_N_STEPS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    sim = phase7._make_source_probe_sim(
        boundary="pec",
        domain=(0.006, 0.006, 0.006),
        dx=0.001,
        cpml_layers=0,
        freq_max=5e9,
    )
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps)
    started = time.perf_counter()
    strategy_a = sim.forward_hybrid_phase1_from_inputs(inputs, strategy="a")
    strategy_b = sim.forward_hybrid_phase1_from_inputs(
        inputs, strategy="b", checkpoint_every=checkpoint_every
    )
    runtime_s = time.perf_counter() - started
    strategy_b_series = np.asarray(strategy_b.time_series).reshape(-1)
    strategy_a_series = np.asarray(strategy_a.time_series).reshape(-1)
    b_peak = extract_resonance_frequency(strategy_b_series, float(inputs.grid.dt))
    a_peak = extract_resonance_frequency(strategy_a_series, float(inputs.grid.dt))
    err = relative_error(b_peak["frequency_hz"], a_peak["frequency_hz"])
    resonance_passed = bool(err <= DEFAULT_RESONANCE_TOLERANCE)
    source_waveforms = [np.asarray(src[-1]).reshape(-1) for src in inputs.raw_sources]
    excitation_series = np.concatenate(source_waveforms) if source_waveforms else None
    bounded = bounded_time_series_proxy(
        strategy_b_series, excitation_series=excitation_series
    )
    return {
        "fixture_id": "source_probe_strategy_b_time_series_small",
        "required": True,
        "runtime_s": round(runtime_s, 6),
        "n_steps": n_steps,
        "checkpoint_every": checkpoint_every,
        "strategy_b_frequency_hz": b_peak["frequency_hz"],
        "observables": {
            "resonance": {
                "observable_source": "strategy_b_time_series",
                "method": b_peak["method"],
                "frequency_hz": b_peak["frequency_hz"],
                "reference_source": "standard_rfx_time_series_reference",
                "reference_frequency_hz": a_peak["frequency_hz"],
                "relative_error": err,
                "tolerance": DEFAULT_RESONANCE_TOLERANCE,
                "sample_count": b_peak["sample_count"],
                "spectral_resolution_hz": b_peak["spectral_resolution_hz"],
                "finite": b_peak["finite"],
                "passed": resonance_passed,
                "failure_reason": None
                if resonance_passed
                else "resonance_tolerance_exceeded",
            },
            "bounded_reflection_proxy": bounded,
            "sparams_reference": {
                "observable_source": "standard_rfx_sparams_reference",
                "strategy_b_seam_ref": "$.strategy_b_seam.native_s_params_supported",
                "applicable": False,
                "not_applicable_reason": (
                    "small Strategy B source/probe fixture has no port S-parameter "
                    "support; passivity gate uses bounded time-series proxy"
                ),
                "max_abs_s11": None,
                "max_abs_s21": None,
                "passivity_tolerance": DEFAULT_PASSIVITY_TOLERANCE,
                "passed": None,
            },
            "input_impedance": input_impedance_check(
                None,
                applicable=False,
                not_applicable_reason=(
                    "no native Strategy B S-parameter/input-impedance observable "
                    "for this fixture; standard-rfx reference recorded separately"
                ),
            ),
        },
    }


def _run_aligned_cavity_strategy_b_fixture(
    *,
    n_steps: int = 512,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    from rfx import GaussianPulse, Simulation

    sim = Simulation(
        freq_max=5e9,
        domain=(ALIGNED_CAVITY_A_M, ALIGNED_CAVITY_B_M, ALIGNED_CAVITY_D_M),
        dx=ALIGNED_CAVITY_DX_M,
        boundary="pec",
        cpml_layers=0,
    )
    sim.add_source(
        (
            ALIGNED_CAVITY_A_M / 3.0,
            ALIGNED_CAVITY_B_M / 3.0,
            ALIGNED_CAVITY_D_M / 2.0,
        ),
        "ez",
        waveform=GaussianPulse(f0=ALIGNED_CAVITY_FREQ_HZ, bandwidth=0.8),
    )
    sim.add_probe(
        (
            2.0 * ALIGNED_CAVITY_A_M / 3.0,
            2.0 * ALIGNED_CAVITY_B_M / 3.0,
            ALIGNED_CAVITY_D_M / 2.0,
        ),
        "ez",
    )
    inputs = sim.build_hybrid_phase1_inputs(n_steps=n_steps)
    started = time.perf_counter()
    strategy_b = sim.forward_hybrid_phase1_from_inputs(
        inputs, strategy="b", checkpoint_every=checkpoint_every
    )
    runtime_s = time.perf_counter() - started
    series = np.asarray(strategy_b.time_series).reshape(-1)
    peak = extract_resonance_frequency(
        series,
        float(inputs.grid.dt),
        f_lo_hz=0.5 * ALIGNED_CAVITY_FREQ_HZ,
        f_hi_hz=1.5 * ALIGNED_CAVITY_FREQ_HZ,
        zero_pad_factor=8,
    )
    err = relative_error(peak["frequency_hz"], ALIGNED_CAVITY_FREQ_HZ)
    passed = bool(err <= DEFAULT_RESONANCE_TOLERANCE)
    source_waveforms = [np.asarray(src[-1]).reshape(-1) for src in inputs.raw_sources]
    excitation_series = np.concatenate(source_waveforms) if source_waveforms else None
    return {
        "fixture_id": "aligned_pec_cavity_tm110_strategy_b_time_series",
        "alignment_target": "tests.test_meep_crossval/tests.test_openems_crossval PEC TM110 cavity",
        "required": False,
        "runtime_s": round(runtime_s, 6),
        "n_steps": n_steps,
        "checkpoint_every": checkpoint_every,
        "grid_shape": list(inputs.grid.shape),
        "strategy_b_frequency_hz": peak["frequency_hz"],
        "analytic_frequency_hz": ALIGNED_CAVITY_FREQ_HZ,
        "observables": {
            "resonance": {
                "observable_source": "strategy_b_time_series",
                "method": peak["method"],
                "frequency_hz": peak["frequency_hz"],
                "reference_source": "analytic_reference",
                "reference_frequency_hz": ALIGNED_CAVITY_FREQ_HZ,
                "relative_error": err,
                "tolerance": DEFAULT_RESONANCE_TOLERANCE,
                "sample_count": peak["sample_count"],
                "spectral_resolution_hz": peak["spectral_resolution_hz"],
                "search_band_hz": peak["search_band_hz"],
                "finite": peak["finite"],
                "passed": passed,
                "failure_reason": None
                if passed
                else "aligned_cavity_resonance_tolerance_exceeded",
            },
            "bounded_reflection_proxy": bounded_time_series_proxy(
                series, excitation_series=excitation_series
            ),
            "input_impedance": input_impedance_check(
                None,
                applicable=False,
                not_applicable_reason=(
                    "aligned PEC cavity validates time-series resonance only; "
                    "native Strategy B port/input-impedance seam is deferred"
                ),
            ),
        },
    }


def build_phase14_artifact(
    *,
    phase13_baseline: Path = DEFAULT_PHASE13_BASELINE,
    external_solvers: Iterable[str] = (),
    require_external_solver: bool = False,
    execute_workload: bool = False,
    n_steps: int = DEFAULT_N_STEPS,
    solver_python: str | None = None,
    solver_pythonpath: str | None = None,
    solver_timeout_s: float | None = None,
) -> dict[str, Any]:
    worktree = worktree_signature()
    baseline = phase13_baseline_record(phase13_baseline)
    seam = strategy_b_seam_record()
    effective_n_steps = max(n_steps, 512) if execute_workload else n_steps
    fixture = _run_default_strategy_b_fixture(n_steps=effective_n_steps)
    fixture["run_class"] = "explicit_workload" if execute_workload else "default_pr_safe"

    requested_solvers = tuple(dict.fromkeys(s.lower() for s in external_solvers))
    fixtures = [fixture]
    if requested_solvers:
        solver_fixture = _run_aligned_cavity_strategy_b_fixture(
            n_steps=max(n_steps, 512)
        )
        solver_fixture["run_class"] = "external_solver_aligned"
        fixtures.append(solver_fixture)
        solver_records = [
            _run_optional_solver_correlation(
                solver,
                required=require_external_solver,
                fixture=solver_fixture,
                solver_python=solver_python,
                solver_pythonpath=solver_pythonpath,
                solver_timeout_s=solver_timeout_s,
            )
            for solver in requested_solvers
        ]
    else:
        solver_records = [
            skipped_solver_record(
                solver,
                required=False,
                reason="not_requested",
                strategy_b_frequency_hz=fixture["strategy_b_frequency_hz"],
            )
            for solver in OPTIONAL_SOLVERS
        ]

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_contract": BENCHMARK_CONTRACT,
        "generated_at": utc_now(),
        "command": ["scripts/phase14_strategy_b_rf_observable_validation.py"],
        "git": {
            "head": worktree["head"],
            "worktree_clean": not worktree["dirty"],
            "status_sha256": worktree["status_sha256"],
        },
        "environment": environment_summary(),
        "phase13_baseline": baseline,
        "strategy_b_seam": seam,
        "fixtures": fixtures,
        "optional_solver_correlation": solver_records,
        "provenance": {
            "execute_workload": execute_workload,
            "external_solvers_requested": list(requested_solvers),
            "require_external_solver": require_external_solver,
            "solver_python": solver_python,
            "solver_pythonpath": solver_pythonpath,
            "solver_timeout_s": solver_timeout_s,
            "worktree_signature": worktree,
        },
    }
    evaluation = evaluate_artifact_gates(artifact)
    artifact.update(evaluation)
    artifact["artifact_id"] = stable_json_hash(
        {k: v for k, v in artifact.items() if k != "artifact_id"}
    )
    return artifact


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_artifact(artifact: Mapping[str, Any], output: Path, *, indent: int | None) -> None:
    output = repo_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=indent, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase13-baseline", type=Path, default=DEFAULT_PHASE13_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--external-solver",
        action="append",
        choices=OPTIONAL_SOLVERS,
        default=[],
        help="Optional external solver correlation lane to run when installed.",
    )
    parser.add_argument(
        "--require-external-solver",
        action="store_true",
        help="Fail closed if a requested external solver is absent or fails.",
    )
    parser.add_argument(
        "--execute-workload",
        action="store_true",
        help="Mark this as an explicit production/full workload evidence run.",
    )
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument(
        "--solver-python",
        help="Optional Python executable for subprocess-isolated Meep runs.",
    )
    parser.add_argument(
        "--solver-pythonpath",
        help="Optional PYTHONPATH prefix for subprocess-isolated solver runs.",
    )
    parser.add_argument(
        "--solver-timeout-s",
        type=float,
        default=None,
        help="Optional timeout for subprocess-isolated solver runs.",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = build_phase14_artifact(
        phase13_baseline=repo_path(args.phase13_baseline),
        external_solvers=args.external_solver,
        require_external_solver=args.require_external_solver,
        execute_workload=args.execute_workload,
        n_steps=args.n_steps,
        solver_python=args.solver_python,
        solver_pythonpath=args.solver_pythonpath,
        solver_timeout_s=args.solver_timeout_s,
    )
    cli_args = sys.argv[1:] if argv is None else argv
    artifact["command"] = [
        "scripts/phase14_strategy_b_rf_observable_validation.py",
        *cli_args,
    ]
    artifact["artifact_id"] = stable_json_hash(
        {k: v for k, v in artifact.items() if k != "artifact_id"}
    )
    write_artifact(artifact, args.output, indent=args.indent)
    print(json.dumps({"output": str(args.output), **artifact["gates"]}, indent=2))
    return 0 if artifact["overall_status"] == "rf_observable_validated_limited" else 1


if __name__ == "__main__":
    raise SystemExit(main())
