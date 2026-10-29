from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTICS_DIR = REPO_ROOT / "scripts" / "diagnostics"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(DIAGNOSTICS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(DIAGNOSTICS_DIR))
        except ValueError:
            pass


floquet_modal_report = _load_module(
    "report_floquet_modal_oracles",
    DIAGNOSTICS_DIR / "report_floquet_modal_oracles.py",
)


def test_floquet_modal_oracle_report_passes_synthetic_te_checks():
    report = floquet_modal_report.evaluate_floquet_modal_oracles()

    assert report["status"] == "passed"
    assert report["claim_scope"].startswith("Floquet/Bloch specular TE")
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["s11_matches_synthetic_gamma"]["status"] == "passed"
    assert checks["s21_matches_synthetic_tau"]["status"] == "passed"
    assert checks["lossless_synthetic_power_balance"]["status"] == "passed"
