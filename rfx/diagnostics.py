"""``rfx-diagnose`` — a one-command install / environment checker for rfx.

Run it after installing rfx to confirm the environment can actually run a
simulation::

    rfx-diagnose

(or ``python -m rfx.diagnostics``). It prints a readable PASS / WARN / FAIL
report and exits ``0`` only when every *critical* check passes. WARN-level
findings (missing optional extras, CPU-only backend, x64 disabled) are
informational and never change the exit code.

The final check is a tiny end-to-end FDTD smoke run — the real "can rfx
actually step a grid on this machine" gate — kept well under ~2 seconds.
"""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata

# ---------------------------------------------------------------------------
# Report primitives
# ---------------------------------------------------------------------------

# Status levels. CRITICAL governs the exit code; WARN/INFO never do.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"


class _Report:
    """Accumulates check lines and tracks whether any critical check failed."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.critical_failures = 0

    def record(self, status: str, label: str, detail: str, *, critical: bool) -> None:
        if status == FAIL and critical:
            self.critical_failures += 1
        # Right-pad the status so the labels line up in the printed report.
        self.lines.append(f"  [{status:<4}] {label}: {detail}")

    def __str__(self) -> str:
        return "\n".join(self.lines)


def _version_of(dist: str) -> str:
    """Best-effort installed-distribution version string ("?" if unknown)."""
    try:
        return metadata.version(dist)
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Individual checks. Each is wrapped so one failure never aborts the rest:
# a check that raises is reported as a FAIL line, not a traceback.
# ---------------------------------------------------------------------------

def _check_python(report: _Report) -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro} ({platform.python_implementation()})"
    # pyproject requires-python = ">=3.10".
    if (v.major, v.minor) >= (3, 10):
        report.record(PASS, "Python version", detail, critical=True)
    else:
        report.record(
            FAIL,
            "Python version",
            f"{detail} — rfx requires Python >= 3.10",
            critical=True,
        )


def _check_rfx_import(report: _Report) -> None:
    try:
        import rfx

        version = getattr(rfx, "__version__", "?")
        location = getattr(rfx, "__file__", "?")
        report.record(
            PASS,
            "import rfx",
            f"rfx {version} (from {location})",
            critical=True,
        )
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        report.record(
            FAIL,
            "import rfx",
            f"could not import rfx: {exc!r}",
            critical=True,
        )


def _check_jax(report: _Report) -> None:
    try:
        import jax

        jax_v = _version_of("jax")
        jaxlib_v = _version_of("jaxlib")
        report.record(
            PASS,
            "jax / jaxlib",
            f"jax {jax_v}, jaxlib {jaxlib_v}",
            critical=True,
        )
    except Exception as exc:
        report.record(
            FAIL,
            "jax / jaxlib",
            f"could not import jax: {exc!r}",
            critical=True,
        )
        return

    # Backend / devices. A GPU backend is reported as PASS; CPU-only is a WARN
    # (large 3D runs will be slow) but never a failure.
    try:
        devices = jax.devices()
        platforms = sorted({d.platform for d in devices})
        device_repr = ", ".join(str(d) for d in devices)
        if any(p in ("gpu", "cuda", "rocm") for p in platforms):
            report.record(
                PASS,
                "jax backend",
                f"GPU backend present: {device_repr}",
                critical=False,
            )
        else:
            report.record(
                WARN,
                "jax backend",
                f"CPU-only ({device_repr}); large 3D runs will be slow "
                f"(not a failure)",
                critical=False,
            )
    except Exception as exc:
        report.record(
            WARN,
            "jax backend",
            f"could not query jax.devices(): {exc!r}",
            critical=False,
        )

    # x64 precision. rfx's S-parameter DFT accumulators run at reduced
    # precision when x64 is disabled — a real rfx caveat, so WARN if off.
    try:
        x64 = bool(jax.config.read("jax_enable_x64"))
        if x64:
            report.record(PASS, "jax x64", "enabled (float64 available)", critical=False)
        else:
            report.record(
                WARN,
                "jax x64",
                "disabled — S-parameter DFT runs reduced precision; "
                "enable with JAX_ENABLE_X64=1 or "
                "jax.config.update('jax_enable_x64', True)",
                critical=False,
            )
    except Exception as exc:
        report.record(WARN, "jax x64", f"could not read config: {exc!r}", critical=False)


def _check_core_deps(report: _Report) -> None:
    # (import name, distribution name). All are hard rfx dependencies, so a
    # missing one is CRITICAL.
    core = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("h5py", "h5py"),
    ]
    for import_name, dist_name in core:
        try:
            importlib.import_module(import_name)
            report.record(
                PASS,
                f"core dep {import_name}",
                _version_of(dist_name),
                critical=True,
            )
        except Exception as exc:
            report.record(
                FAIL,
                f"core dep {import_name}",
                f"missing or broken: {exc!r}",
                critical=True,
            )


def _check_optional_extras(report: _Report) -> None:
    # Optional extras: report presence only, never fail. These back the
    # [optimization] / [dashboard] / [visualization] extras.
    optional = [
        ("optax", "optax", "inverse-design / gradient optimisation"),
        ("streamlit", "streamlit", "dashboard UI"),
        ("pandas", "pandas", "tabular post-processing"),
        ("PIL", "pillow", "image export"),
    ]
    for import_name, dist_name, purpose in optional:
        try:
            importlib.import_module(import_name)
            report.record(
                INFO,
                f"optional {dist_name}",
                f"{_version_of(dist_name)} ({purpose})",
                critical=False,
            )
        except Exception:
            report.record(
                INFO,
                f"optional {dist_name}",
                f"not installed ({purpose}) — install only if needed",
                critical=False,
            )


def _check_smoke(report: _Report) -> None:
    """Tiny end-to-end FDTD run — the real "can rfx run here" gate.

    Builds a few-cell uniform PEC box, adds one point source and one point
    probe, steps a handful of timesteps, and asserts the probe time series is
    finite. CRITICAL. Kept well under ~2 s.
    """
    try:
        import numpy as np

        from rfx import Simulation
        from rfx.sources.sources import GaussianPulse

        # ~10-cell-per-axis box (dx = 2 mm over a 20 mm domain). PEC boundary
        # keeps it cheap — no CPML profiles to build.
        sim = Simulation(
            freq_max=10e9,
            domain=(0.02, 0.02, 0.02),
            dx=2e-3,
            boundary="pec",
        )
        sim.add_source(
            (0.01, 0.01, 0.01),
            "ez",
            waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
        )
        sim.add_probe((0.012, 0.01, 0.01), "ez")

        result = sim.run(n_steps=40, compute_s_params=False, skip_preflight=True)
        ts = np.asarray(result.time_series)
        finite = bool(np.all(np.isfinite(ts)))
        grid_repr = (
            tuple(int(n) for n in result.grid.shape)
            if result.grid is not None
            else "n/a"
        )

        if finite:
            report.record(
                PASS,
                "FDTD smoke run",
                f"grid {grid_repr}, 40 steps, "
                f"finite probe trace (peak |Ez|={float(np.max(np.abs(ts))):.3e})",
                critical=True,
            )
        else:
            report.record(
                FAIL,
                "FDTD smoke run",
                "probe time series contained NaN/Inf — the run diverged",
                critical=True,
            )
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        report.record(
            FAIL,
            "FDTD smoke run",
            f"simulation failed to run: {exc!r}",
            critical=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run all diagnostics and return 0 iff every critical check passed."""
    # argv is accepted for the console-script / test contract; there are no
    # options to parse yet, so it is intentionally unused.
    del argv

    report = _Report()

    print("rfx environment diagnostics")
    print("=" * 60)

    # Order matters: import rfx and core deps before the smoke run so a
    # missing dependency is reported as its own clean FAIL line rather than
    # surfacing only as a smoke-run traceback.
    _check_python(report)
    _check_rfx_import(report)
    _check_jax(report)
    _check_core_deps(report)
    _check_optional_extras(report)
    _check_smoke(report)

    print(report)
    print("=" * 60)

    if report.critical_failures == 0:
        print("Summary: All critical checks passed.")
        return 0
    n = report.critical_failures
    print(f"Summary: {n} critical check(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
