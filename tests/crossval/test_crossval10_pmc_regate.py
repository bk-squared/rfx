"""cv10 re-gate falsifiers (issue #812, critical tier).

Issue #812 measured that monkeypatching ``apply_pmc_faces`` to a no-op —
deleting the boundary condition cv10 is *named for* — makes cv10's gate
score BETTER: the uniform peak spread improves from 0.2487 % to 0.0667 %
against a 2 % gate, with the peaks scaled by 0.5077 and the field
bit-identical to a PEC wall.

The mechanism is that gates 1/2 are within-path RELATIVE spreads, and
``(peak_max − peak_min)/peak_max`` is invariant under
``peak_i → c·peak_i``. Deleting a boundary condition on a face the direct
source→probe path never touches is exactly such a constant factor.

These tests pin the two gates that repair it, and — crucially — pin the
falsifier itself, so a future refactor cannot quietly return cv10 to a
state where deleting the wall is undetectable:

  * gate 3, PMC realization: ``H_tan == 0.0`` bit-exact on the declared
    y_lo face (definitional threshold — ``apply_pmc_faces`` writes the
    literal ``0.0``), with a non-degeneracy guard;
  * gate 4, image-doubling control arm: the half-domain PMC amplitude
    must reproduce the mirrored full-domain image-source amplitude to
    within 2 % — the ABSOLUTE check the relative spread threw away.

Thresholds and the control geometry were frozen in
``docs/design_notes/cv10_pmc_realization_regate.md`` (commit a00a53d)
before any judging measurement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV10_PATH = REPO_ROOT / "validation" / "crossval" / "10_pmc_cpml_half_symmetric.py"


def _load_cv10():
    spec = importlib.util.spec_from_file_location("_cv10_pmc_regate", CV10_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cv10 = _load_cv10()

PATHS = ("uniform", "nonuniform")


# ---------------------------------------------------------------------------
# Geometry — pure arithmetic, no solve. The control domain must follow from
# the stated half-cell PMC-plane convention, not from a hand-typed number.
# ---------------------------------------------------------------------------


def test_control_geometry_follows_from_the_pmc_plane_convention():
    dx = cv10.DX
    # apply_pmc_faces zeros H_tan at array index 0 => plane 0.5*dx inside.
    assert cv10.MIRROR_PLANE_Y == pytest.approx(0.5 * dx)
    # Ez is even about the plane, so node y' images onto 2*plane - y'.
    assert cv10.FULL_IMAGE_Y - cv10.Y_SHIFT == pytest.approx(
        2 * cv10.MIRROR_PLANE_Y - cv10.SRC_Y)
    # Interior nodes 1..20 mm image onto 0..-19 mm => full extent 39 mm.
    assert cv10.FULL_DOM[1] == pytest.approx(39e-3)
    assert cv10.FULL_DOM[0] == cv10.DOM[0] and cv10.FULL_DOM[2] == cv10.DOM[2]
    # Plane sits equidistant from both ends of the control domain.
    plane_full = cv10.MIRROR_PLANE_Y + cv10.Y_SHIFT
    assert plane_full == pytest.approx(19.5e-3)
    assert plane_full == pytest.approx(cv10.FULL_DOM[1] - plane_full)
    # Source pair straddles the plane by half a cell each way.
    assert cv10.FULL_SRC_Y == pytest.approx(20e-3)
    assert cv10.FULL_IMAGE_Y == pytest.approx(19e-3)
    assert cv10.FULL_PROBE_Y == pytest.approx(24e-3)
    # The window was frozen before measurement; never widen it.
    assert cv10.IMAGE_TOL == 0.02
    assert cv10.CONTROL_CPML == max(cv10.CPML_VALUES)


# ---------------------------------------------------------------------------
# Criterion (A): the case still passes on today's correct code.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def half_runs():
    return {p: cv10._run_half(cv10.CONTROL_CPML, p) for p in PATHS}


@pytest.fixture(scope="module")
def full_runs():
    return {p: cv10.run_full_image(p) for p in PATHS}


@pytest.mark.parametrize("path", PATHS)
def test_gate3_pmc_face_is_realized_bit_exact(path, half_runs):
    r = half_runs[path]
    assert r["face_hx"] == 0.0, (
        f"{path}: PMC y_lo did not zero Hx — max|Hx[:,0,:]| = {r['face_hx']:.6e}")
    assert r["face_hz"] == 0.0, (
        f"{path}: PMC y_lo did not zero Hz — max|Hz[:,0,:]| = {r['face_hz']:.6e}")
    # Non-degeneracy: the zero must be a boundary condition, not an
    # unexcited field.
    assert r["dom_h"] > 0.0, f"{path}: no H field anywhere — source never fired"
    assert r["peak"] > 0.0


@pytest.mark.parametrize("path", PATHS)
def test_gate4_half_domain_reproduces_full_domain_amplitude(path, half_runs,
                                                            full_runs):
    ratio, ok = cv10.evaluate_image_control(half_runs[path]["peak"],
                                            full_runs[path]["peak"])
    assert ok, (
        f"{path}: half-domain PMC amplitude does not reproduce the mirrored "
        f"full-domain image-source amplitude: R = {ratio:.6f}, "
        f"|R−1| = {abs(ratio - 1)*100:.4f} % >= {cv10.IMAGE_TOL*100:.0f} %")


# ---------------------------------------------------------------------------
# Criterion (B): the new gates must FAIL on the defect #812 measured them
# blind to, and the old gate must be shown still passing on it — that
# contrast is the whole finding.
# ---------------------------------------------------------------------------


def test_noop_pmc_defeats_the_old_relative_gate_and_is_caught_by_the_new_ones(
        monkeypatch, full_runs):
    """Delete the boundary condition; the relative spread still passes."""
    import rfx.boundaries.pmc as pmc_mod

    monkeypatch.setattr(pmc_mod, "apply_pmc_faces",
                        lambda state, faces: state)

    lo = cv10._run_half(min(cv10.CPML_VALUES), "uniform")
    hi = cv10._run_half(cv10.CONTROL_CPML, "uniform")

    peaks = (lo["peak"], hi["peak"])
    spread = (max(peaks) - min(peaks)) / max(peaks)

    # --- the finding: gate 1 is BLIND (it still passes, in fact better) ---
    assert spread < 0.02, (
        "expected the relative spread to remain inside its 2 % gate with the "
        f"wall deleted (that is the #812 finding); got {spread*100:.4f} %")

    # --- gate 3 catches it: the face is no longer a magnetic wall ---
    worst_face = max(hi["face_hx"], hi["face_hz"])
    assert worst_face > 0.0, (
        "gate 3 did not detect the deleted PMC: max|H_tan| on y_lo is still "
        "exactly 0.0 with apply_pmc_faces monkeypatched to a no-op")

    # --- gate 4 catches it: the image sign flipped, so the amplitude did ---
    ratio, ok = cv10.evaluate_image_control(hi["peak"],
                                            full_runs["uniform"]["peak"])
    assert not ok, (
        f"gate 4 did not detect the deleted PMC: R = {ratio:.6f} is still "
        f"inside |R−1| < {cv10.IMAGE_TOL*100:.0f} %")
    # PEC-image cancellation, not a small perturbation.
    assert ratio < 0.75, (
        f"expected a large absolute amplitude loss from the flipped image "
        f"sign; got R = {ratio:.6f}")
