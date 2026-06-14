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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx import Simulation

MATRIX_PATH = Path("docs/guides/sparameter_support_matrix.json")

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
