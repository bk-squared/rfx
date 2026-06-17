"""AD-surface contract (roadmap W4.6).

Every public S-parameter-shaped entry point — top-level ``rfx`` exports
matching ``compute_*``/``extract_*`` plus ``Simulation.compute_*`` methods —
must carry an explicit autodiff classification:

- ``grad-safe``     : a named grad smoke/end-to-end test exists (pointer kept
                      honest by ``test_grad_safe_evidence_pointers_exist``).
- ``not-traceable`` : numpy / concretizing by design (diagnostic post-
                      processors); the load-bearing coaxial case is verified
                      EMPIRICALLY below, not by prose.
- ``untested``      : no grad evidence either way; must cite a tracking
                      pointer so it cannot silently rot.

The point of the contract: exporting a new ``compute_*``/``extract_*`` without
deciding its AD story fails ``test_every_sparam_entry_point_is_ad_classified``.
The per-port-family view of the same information lives in
``docs/guides/sparameter_support_matrix.json`` (``ad_traceable`` column),
locked by ``test_support_matrix_has_ad_traceable_column``.

These tests do not run FDTD.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx import Simulation

_REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = _REPO_ROOT / "docs/guides/sparameter_support_matrix.json"
MANIFEST_PATH = _REPO_ROOT / "scripts/diagnostics/port_external_reference_requirements.json"

GRAD_SAFE = "grad-safe"
NOT_TRACEABLE = "not-traceable"
UNTESTED = "untested"

# name -> (classification, evidence)
# GRAD_SAFE evidence is "path::test[, path::test...]" and is existence-checked.
AD_CLASSIFICATION = {
    # --- Simulation methods -------------------------------------------------
    "Simulation.compute_waveguide_s_matrix": (
        GRAD_SAFE,
        "tests/test_sparam_ad_end_to_end.py::test_waveguide_s_matrix_ad_end_to_end, "
        "tests/test_waveguide_sparam_ad.py::test_wg_smatrix_assembly_ad_traceable",
    ),
    "Simulation.compute_msl_s_matrix": (
        GRAD_SAFE,
        "tests/test_msl_sparam_ad.py::test_compute_msl_s_matrix_ad_smoke_has_finite_gradient, "
        "tests/test_sparam_ad_end_to_end.py::test_msl_s_matrix_ad_end_to_end",
    ),
    "Simulation.compute_coaxial_s_matrix": (
        NOT_TRACEABLE,
        "deprecated single-plane V/I path; numpy extraction, no differentiable input",
    ),
    "Simulation.compute_coaxial_line_reflection": (
        NOT_TRACEABLE,
        "numpy matrix-pencil extraction; verified empirically by "
        "test_coaxial_reflection_extraction_breaks_tape in this file",
    ),
    # --- top-level exports --------------------------------------------------
    "compute_error_indicator": (
        NOT_TRACEABLE,
        "numpy AMR diagnostic (rfx/amr.py)",
    ),
    "compute_far_field": (
        NOT_TRACEABLE,
        "numpy NTFF post-processor; the AD path is compute_far_field_jax",
    ),
    "compute_far_field_jax": (
        GRAD_SAFE,
        "tests/test_directivity_gradient.py::test_maximize_directivity_gradient_nonzero",
    ),
    "compute_rcs": (
        NOT_TRACEABLE,
        "numpy RCS post-processor on top of compute_far_field",
    ),
    "compute_floquet_s_params": (
        GRAD_SAFE,
        "first real pytest caller + finite-grad smoke (issue #141): "
        "tests/test_floquet_s_params_contract.py::test_compute_floquet_s_params_grad_finite, "
        "tests/test_floquet_s_params_contract.py::test_compute_floquet_s_params_real_fdtd_runs — "
        "https://github.com/bk-squared/rfx/issues/141",
    ),
    "extract_floquet_modes": (
        GRAD_SAFE,
        "finite-grad smoke through the jnp-pure mode extractor (issue #141): "
        "tests/test_floquet_s_params_contract.py::test_extract_floquet_modes_grad_finite — "
        "https://github.com/bk-squared/rfx/issues/141",
    ),
    "extract_coaxial_plane_vi_from_dft": (
        NOT_TRACEABLE,
        "numpy V/I diagnostic extractor (rfx/sources/coaxial_port.py)",
    ),
    "extract_multimode_s_matrix": (
        NOT_TRACEABLE,
        "concretizing casts in the multimode assembly (rfx/sources/waveguide_port.py)",
    ),
    "extract_s_matrix_wire": (
        NOT_TRACEABLE,
        "numpy run()-path post-processor; the wire-port AD path is "
        "forward(port_s11_freqs=...) — tests/test_wire_port_sparams_forward.py",
    ),
    "extract_waveguide_s_matrix": (
        GRAD_SAFE,
        "exercised inside Simulation.compute_waveguide_s_matrix: "
        "tests/test_sparam_ad_end_to_end.py::test_waveguide_s_matrix_ad_end_to_end",
    ),
    "extract_waveguide_port_waves": (
        UNTESTED,
        "jnp-pure helper, no direct grad smoke test",
    ),
    "extract_waveguide_sparams": (
        UNTESTED,
        "jnp-pure legacy helper, no direct grad smoke test",
    ),
    "extract_waveguide_s11": (
        UNTESTED,
        "jnp-pure legacy helper, no direct grad smoke test",
    ),
    "extract_waveguide_s21": (
        UNTESTED,
        "jnp-pure legacy helper, no direct grad smoke test",
    ),
}


def _exported_surface() -> set[str]:
    names = {
        n
        for n in dir(rfx)
        if n.startswith(("compute_", "extract_")) and callable(getattr(rfx, n))
    }
    for n, _ in inspect.getmembers(Simulation, predicate=inspect.isfunction):
        if n.startswith("compute_"):
            names.add(f"Simulation.{n}")
    return names


def test_every_sparam_entry_point_is_ad_classified():
    surface = _exported_surface()
    missing = sorted(surface - AD_CLASSIFICATION.keys())
    stale = sorted(AD_CLASSIFICATION.keys() - surface)
    assert not missing, (
        "New compute_*/extract_* entry points exported without an AD "
        f"classification: {missing}. Add each to AD_CLASSIFICATION in this "
        "file (grad-safe with a named grad test, not-traceable with the "
        "reason, or untested with a tracking pointer) and mirror the port-"
        "family view in docs/guides/sparameter_support_matrix.json."
    )
    assert not stale, (
        f"AD_CLASSIFICATION entries no longer exported: {stale}. Remove them."
    )


def test_grad_safe_evidence_pointers_exist():
    for name, (status, evidence) in AD_CLASSIFICATION.items():
        if status != GRAD_SAFE:
            continue
        refs = re.findall(r"(tests/[\w/]+\.py)::(\w+)", evidence)
        assert refs, f"{name}: grad-safe entry must cite at least one path::test"
        for path, testname in refs:
            p = Path(path)
            assert p.exists(), f"{name}: evidence file {path} is gone"
            assert f"def {testname}" in p.read_text(), (
                f"{name}: evidence test {path}::{testname} no longer exists"
            )


def test_coaxial_reflection_extraction_breaks_tape():
    """Empirical basis for the coaxial ``not-traceable`` classification.

    The roadmap critic flagged the earlier "coax breaks the tape" claim as
    digest prose; this test IS the evidence. A synthetic two-wave voltage
    profile with planted A=theta, B=0.3 must (a) give the planted |B/A| on the
    concrete path (witness that we exercise the real extractor) and (b) raise
    TracerArrayConversionError under jax.grad, because
    ``coaxial_line_reflection_from_plane_voltages`` is numpy (np.asarray /
    np.linalg.lstsq / complex()). If (b) starts succeeding, the extraction
    became traceable — upgrade the classification instead of deleting this.
    """
    from rfx.sources.coaxial_port import coaxial_line_reflection_from_plane_voltages

    z = np.linspace(0.002, 0.013, 12)

    def objective(theta):
        V = theta * jnp.exp(1j * 300.0 * jnp.asarray(z)) + 0.3 * jnp.exp(
            -1j * 300.0 * jnp.asarray(z)
        )
        res = coaxial_line_reflection_from_plane_voltages(
            z, V, reference_plane_m=0.0
        )
        return jnp.abs(jnp.asarray(res.reflection))

    concrete = float(objective(jnp.asarray(0.7)))
    assert abs(concrete - 0.3 / 0.7) < 1e-3, (
        f"concrete witness drifted: |Gamma|={concrete:.6f}, expected ~{0.3/0.7:.6f}"
    )
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.grad(objective)(jnp.asarray(0.7))


def _collect_nodeids(paths, marker_filter=None):
    """COLLECTION-TIME nodeid set for the given test files (T2.2).

    Uses a child ``pytest --collect-only`` (NOT a regex/AST scrape of the
    source — m1) and clears the repo's default ``addopts`` (which deselects
    ``gpu``/``slow``) so a gpu-marked AD test is still collected; only the
    explicit ``marker_filter`` is applied. With ``marker_filter='not (xfail or
    skip or skipif)'`` an xfail/skip/skipif-marked test is dropped, so comparing
    the two sets reveals "exists but cannot pass".
    """
    args = [sys.executable, "-m", "pytest", "--collect-only", "-q",
            "-o", "addopts=", "-p", "no:cacheprovider"]
    if marker_filter:
        args += ["-m", marker_filter]
    args += list(paths)
    proc = subprocess.run(args, capture_output=True, text=True, cwd=_REPO_ROOT)
    nodeids = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        # pytest -q --collect-only emits one nodeid per line: ``path::name``.
        if re.match(r"^\S+\.py::\S+$", line):
            nodeids.add(line)
    return nodeids, proc


def _nodeid_present(nodeid, collected):
    return nodeid in collected or any(
        c == nodeid or c.startswith(nodeid + "[") for c in collected
    )


def test_ad_fd_gate_tests_are_collected_and_not_xfail_skip():
    """T2.2: every declared `ad_fd_test` must be COLLECTED and NOT xfail/skip.

    The auditor (`check_port_external_references.py`) statically requires a
    declared+existing `ad_fd_test` for broad_e5_passed; existence is not enough
    (the framework audit's gameability lesson). This test is the dynamic half:
    a declared AD-vs-FD test that is xfail/skip-marked CANNOT satisfy the moat,
    so it must be collected under `not (xfail or skip or skipif)`. gpu-marked
    tests pass (they run on the gpu lane, not xfail/skip).

    Boundary (documented, not enforced here): this proves "collected + not
    statically xfail/skip/skipif-marked", NOT "executed green". A test that calls
    `pytest.skip()` in its body, or a gpu-only test whose hardware lane has not
    run this cycle, is collectable + unmarked yet may not have actually exercised
    the FD comparison. The gpu lane (e.g. the MSL AD-vs-FD test) is run per the
    release cadence in CLAUDE.md; this gate guarantees the pointer is real and
    expected-to-pass, the lane guarantees it did.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    declared = [
        (e["family"], e["ad_fd_test"])
        for e in manifest["requirements"]
        if e.get("ad_fd_test")
    ]
    assert declared, "no family declares an ad_fd_test — manifest regression"

    paths = sorted({nodeid.split("::", 1)[0] for _, nodeid in declared})
    all_collected, proc_all = _collect_nodeids(paths)
    # pytest --collect-only exit codes: 0 = collected, 5 = none collected.
    assert proc_all.returncode in (0, 5), (
        f"pytest --collect-only failed (rc={proc_all.returncode}); "
        f"stderr:\n{proc_all.stderr[-2000:]}"
    )
    assert all_collected, (
        "pytest --collect-only returned no nodeids for the AD-vs-FD test files; "
        f"stderr:\n{proc_all.stderr[-2000:]}"
    )
    passable, proc_pass = _collect_nodeids(
        paths, marker_filter="not (xfail or skip or skipif)"
    )
    assert proc_pass.returncode in (0, 5) and passable, (
        "marker-filtered collection returned nothing — parse/infra error, not a "
        f"real xfail/skip verdict; stderr:\n{proc_pass.stderr[-2000:]}"
    )

    for family, nodeid in declared:
        assert _nodeid_present(nodeid, all_collected), (
            f"{family}: ad_fd_test {nodeid} is not collected (typo / removed / "
            f"renamed). Update the manifest or restore the test."
        )
        assert _nodeid_present(nodeid, passable), (
            f"{family}: ad_fd_test {nodeid} is xfail/skip/skipif-marked, so it "
            f"cannot prove the differentiability moat. An AD-vs-FD gate must be a "
            f"test expected to PASS (gpu-marked is fine)."
        )


def test_support_matrix_has_ad_traceable_column():
    data = json.loads(MATRIX_PATH.read_text())
    allowed = {"yes", "no", "untested", "not_applicable"}
    for fam in data["port_families"]:
        name = fam["family"]
        assert "ad_traceable" in fam, f"{name}: missing ad_traceable"
        assert fam["ad_traceable"] in allowed, (
            f"{name}: ad_traceable={fam['ad_traceable']!r} not in {sorted(allowed)}"
        )
        if fam["ad_traceable"] in {"yes", "no", "untested"}:
            assert fam.get("ad_evidence"), (
                f"{name}: ad_traceable={fam['ad_traceable']} requires ad_evidence"
            )
