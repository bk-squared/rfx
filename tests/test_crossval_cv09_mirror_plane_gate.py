"""Falsifier battery for cv09's re-gated mirror plane (issue #812, Phase 1).

Issue #812 measured that ``validation/crossval/09_half_symmetric_waveguide.py``
could not fail for its own stated reason:

* the ``a/2 -> a/2 + dx/2`` half-cell declaration change PR #762 made is a
  **no-op at this mesh** — both declarations build grid (24, 21, 61) and
  realize the same mirror plane — so the case's headline quantity was never
  gated at all; and
* gate 3's 5 % window was 1.7x wider than the 2.70-3.00 % signature of a
  one-cell mirror-plane error and 2.7x wider than the 1.72-1.84 % signature
  of a half-cell one.

The re-gate adds gate 0 on the REALIZED mirror plane
(``a_eff = 2 * (realized H_tan wall)``, tolerance ``DX/4``) and tightens gate 3
to that budget's frequency image, ``(d^2/(a^2+d^2))*(DX/4)/a = 0.3556 %``.
Thresholds were pre-declared in ``docs/design_notes/
issue812_cv09_mirror_plane_regate.md`` before any measurement judged them.

Every test below drives cv09's OWN gate functions (``geometry_gate``,
``G3_TOL``, ``_run_cavity``, ``_extract_mode_near``) — none re-implements the
gate math, so deleting or loosening a gate in the script reds this file.

The geometry tests build grids only (milliseconds, no FDTD). Three short CPU
solves at the end carry the frequency-domain half of the falsifier.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
CV09_PATH = REPO_ROOT / "validation" / "crossval" / "09_half_symmetric_waveguide.py"


def _load_cv09():
    """Import cv09 without executing its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location("_cv09_mirror_gate", CV09_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cv09 = _load_cv09()


def _half_axes(half_x: float, dx: float | None = None) -> dict:
    """Realized-geometry rows for a PEC+PMC half cavity declared at
    ``half_x``, built through cv09's own reporter reader."""
    from rfx import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    sim = Simulation(
        freq_max=cv09.FREQ_MAX,
        domain=(half_x, cv09.b, cv09.d),
        dx=cv09.DX if dx is None else dx,
        boundary=BoundarySpec(
            x=Boundary(lo="pec", hi="pmc"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=0,
    )
    return cv09.realized_axes(sim)


def _full_axes(dx: float | None = None) -> dict:
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec

    sim = Simulation(
        freq_max=cv09.FREQ_MAX,
        domain=(cv09.a, cv09.b, cv09.d),
        dx=cv09.DX if dx is None else dx,
        boundary=BoundarySpec.uniform("pec"),
        cpml_layers=0,
    )
    return cv09.realized_axes(sim)


# ----------------------------------------------------------------------
# The defect itself: the #762 declaration change is a no-op at this mesh.
# ----------------------------------------------------------------------

def test_the_762_declaration_change_is_a_noop_at_this_mesh():
    """#812's headline measurement, reproduced: ``a/2`` (pre-#762) and
    ``a/2 + dx/2`` (post-#762) realize the SAME mirror plane at DX = 0.508 mm,
    because ``ceil(22.5) == ceil(23.0) == 23``.

    This is why a gate on the realized plane is the right instrument: it does
    not read the declaration, so it reports the two as identical — correctly —
    instead of crediting a change that carried no information."""
    pre = _half_axes(0.5 * cv09.a)
    post = _half_axes(0.5 * cv09.a + 0.5 * cv09.DX)
    assert pre["x"]["n_cells"] == post["x"]["n_cells"] == 23
    assert cv09.mirror_a_eff(pre) == pytest.approx(cv09.mirror_a_eff(post))
    assert cv09.mirror_a_eff(post) == pytest.approx(cv09.a, abs=1e-12)


def test_a_eff_is_an_odd_multiple_of_dx_so_only_odd_meshes_can_register():
    """The lattice fact ``DX/4`` is derived from: ``a_eff = (2n - 1) * dx``.

    Hence ``a_eff == a`` is reachable only when ``a/dx`` is ODD, and the
    smallest nonzero misregistration is ``dx`` (even ``a/dx``) or ``2*dx``
    (odd ``a/dx``) — never smaller than ``4 * DX/4``."""
    for n_cells, half_x in ((22, 0.5 * cv09.a - 0.5 * cv09.DX),
                            (23, 0.5 * cv09.a + 0.5 * cv09.DX),
                            (24, 0.5 * cv09.a + 1.5 * cv09.DX)):
        axes = _half_axes(half_x)
        assert axes["x"]["n_cells"] == n_cells
        assert cv09.mirror_a_eff(axes) == pytest.approx(
            (2 * n_cells - 1) * cv09.DX, rel=1e-12)


# ----------------------------------------------------------------------
# Criterion (A): the gate passes on the correct declaration.
# ----------------------------------------------------------------------

def test_gate0_passes_on_the_correct_declaration():
    ok, lines = cv09.geometry_gate(
        _full_axes(), _half_axes(0.5 * cv09.a + 0.5 * cv09.DX))
    assert ok, "\n".join(lines)
    assert all(line.startswith("PASS") for line in lines)


# ----------------------------------------------------------------------
# Criterion (B): the gate fails on the defects it was measured blind to.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("sign,expect_cells,expect_a_eff_mm", [
    (+1, 24, 23.876),
    (-1, 22, 21.844),
])
def test_gate0_fires_on_a_one_cell_mirror_plane_error(
        sign, expect_cells, expect_a_eff_mm):
    """A one-cell mirror-plane error — the smallest error the OLD 5 % gate
    tolerated — must fail gate 0 by a wide margin."""
    axes = _half_axes(0.5 * cv09.a + 0.5 * cv09.DX + sign * cv09.DX)
    assert axes["x"]["n_cells"] == expect_cells
    a_eff = cv09.mirror_a_eff(axes)
    assert a_eff * 1e3 == pytest.approx(expect_a_eff_mm, abs=1e-3)
    resid = abs(a_eff - cv09.a)
    assert resid == pytest.approx(2 * cv09.DX, rel=1e-9)
    assert resid / cv09.GEOM_TOL == pytest.approx(8.0, rel=1e-9)  # 1016.0 vs 127.0 um
    ok, lines = cv09.geometry_gate(_full_axes(), axes)
    assert not ok
    assert sum(line.startswith("FAIL") for line in lines) == 1
    assert any(line.startswith("FAIL") and "a_eff" in line for line in lines)


def test_gate0_fires_on_the_pre_762_convention_at_its_own_mesh():
    """The pre-#762 declaration (``a/2``) at the mesh it actually shipped on
    (DX = 0.635 mm, ``a = 36 dx``, EVEN) realizes ``a_eff = a - dx``: the
    half-cell bias the script docstring measured as gate 3 = 1.825 %.

    The same mesh also shows that ``+ dx/2`` is not a fix there either — it
    realizes ``a_eff = a + dx`` (docstring: gate 3 = 1.722 %). On a ceil-based
    grid the declaration term never converts a wrong mirror plane into a right
    one; the odd-cell mesh does. Gate 0 fails on BOTH."""
    dx_old = 0.635e-3
    tol_old = dx_old / 4.0
    for label, half_x, expected in (
            ("pre-#762 a/2", 0.5 * cv09.a, cv09.a - dx_old),
            ("naive a/2+dx/2", 0.5 * cv09.a + 0.5 * dx_old, cv09.a + dx_old)):
        axes = _half_axes(half_x, dx=dx_old)
        a_eff = cv09.mirror_a_eff(axes)
        assert a_eff == pytest.approx(expected, rel=1e-9), label
        assert abs(a_eff - cv09.a) == pytest.approx(dx_old, rel=1e-9), label
        ok, lines = cv09.geometry_gate(_full_axes(dx=dx_old), axes,
                                       tol=tol_old)
        assert not ok, label
        assert any(line.startswith("FAIL") and "a_eff" in line
                   for line in lines), label


def test_gate0_fires_on_the_origin_main_incommensurate_mesh():
    """REALIZE-DECLARED-BY-MESH (#722/#724) on the FULL run: the pre-#762
    origin/main mesh DX = 0.5 mm rasterizes the 22.86 mm broad wall to
    23.000 mm, the mismatch the docstring says "is most of what gate 1 was
    reporting". Gate 0 now says so directly."""
    dx_om = 0.5e-3
    axes = _full_axes(dx=dx_om)
    assert axes["x"]["n_cells"] == 46
    assert axes["x"]["realized_extent"] == pytest.approx(23.0e-3, abs=1e-9)
    ok, lines = cv09.geometry_gate(
        axes, _half_axes(0.5 * cv09.a + 0.5 * dx_om, dx=dx_om),
        tol=dx_om / 4.0)
    assert not ok
    assert any(line.startswith("FAIL") and "full cavity" in line
               for line in lines)


# ----------------------------------------------------------------------
# The old gate's blindness, stated as an assertion.
# ----------------------------------------------------------------------

def test_old_five_percent_gate_was_blind_to_every_defect_above():
    """Pozar-predicted gate-3 readings for each defect, against the OLD 5 %
    window and the NEW G3_TOL. The old window admits all four; the new one
    admits none. (Frequencies from the analytic closed form, so this test is
    the arithmetic statement of the blindness — the FDTD confirmation is
    below.)"""
    def f101(a_eff):
        return 0.5 * cv09.C0 * np.sqrt((1.0 / a_eff) ** 2 + (1.0 / cv09.d) ** 2)

    f0 = f101(cv09.a)
    defects = {
        "one-cell hi": cv09.a + 2 * cv09.DX,
        "one-cell lo": cv09.a - 2 * cv09.DX,
        "pre-#762 half-cell (dx=0.635)": cv09.a - 0.635e-3,
        "naive a/2+dx/2 (dx=0.635)": cv09.a + 0.635e-3,
    }
    for label, a_eff in defects.items():
        dev = abs(f101(a_eff) - f0) / f0
        assert dev < 0.05, f"{label}: OLD 5% gate PASSED at {dev:.3%}"
        assert dev > cv09.G3_TOL, f"{label}: new gate must fail, got {dev:.3%}"
        assert dev > 4 * cv09.G3_TOL, f"{label}: margin too thin ({dev:.3%})"


def test_g3_tol_is_the_frequency_image_of_the_geometry_budget():
    """G3_TOL must stay tied to DX/4 through the Pozar log-derivative, not
    drift into a hand-set literal."""
    sens = cv09.d ** 2 / (cv09.a ** 2 + cv09.d ** 2)
    assert sens == pytest.approx(16.0 / 25.0, rel=1e-9)   # a/d = 3/4 exactly
    assert cv09.G3_TOL == pytest.approx(sens * cv09.GEOM_TOL / cv09.a, rel=1e-12)
    assert cv09.G3_TOL == pytest.approx(3.5556e-3, rel=1e-4)
    assert cv09.G3_TOL < 0.05 / 10          # never widened, >10x tighter


def test_fft_fallback_is_not_in_the_judged_path():
    """#812: the windowed-FFT peak-pick quantises at 1/(3072 dt) = 335.9 MHz
    = 4.099 % of f_101, 11.5x G3_TOL — it cannot judge gate 3. It must not
    exist as a fallback, only as a diagnostic printer."""
    assert not hasattr(cv09, "_fft_peak_near")
    assert hasattr(cv09, "_print_fft_diagnostic")
    src = CV09_PATH.read_text()
    assert '"fft"' not in src and "'fft'" not in src
    # A silent ringdown yields no Harminv mode -> None, not a fabricated peak.
    silent = np.zeros(3072)
    assert cv09._extract_mode_near(silent, 1e-12, cv09.F_101_ANALYTIC) is None


# ----------------------------------------------------------------------
# FDTD leg: the new gate 3 actually fires on an injected one-cell error.
# ----------------------------------------------------------------------

def _solve_freq(domain, spec, n_steps=None):
    ts, dt, axes = cv09._run_cavity(
        domain=domain, spec=spec,
        source_pos=(0.25 * cv09.a, 0.5 * cv09.b, 0.5 * cv09.d),
        probe_pos=(0.40 * cv09.a, 0.5 * cv09.b, 0.33 * cv09.d),
        n_steps=cv09.N_STEPS if n_steps is None else n_steps,
    )
    mode = cv09._extract_mode_near(ts, dt, cv09.F_101_ANALYTIC)
    assert mode is not None, "harminv found no mode"
    return float(mode.freq), axes


def test_gate3_fires_on_an_injected_one_cell_mirror_error_fdtd():
    """Criterion (B) in the frequency domain, measured not asserted: solve the
    full cavity and the one-cell-wrong half cavity and push the pair through
    cv09's own gate-3 comparison."""
    from rfx.boundaries.spec import Boundary, BoundarySpec

    spec_full = BoundarySpec.uniform("pec")
    spec_half = BoundarySpec(
        x=Boundary(lo="pec", hi="pmc"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    )
    f_full, axes_full = _solve_freq((cv09.a, cv09.b, cv09.d), spec_full)

    half_ok = 0.5 * cv09.a + 0.5 * cv09.DX
    f_ok, axes_ok = _solve_freq((half_ok, cv09.b, cv09.d), spec_half)
    gap_ok = abs(f_full - f_ok) / f_full
    assert gap_ok < cv09.G3_TOL, f"criterion (A) regressed: {gap_ok:.4%}"
    assert cv09.geometry_gate(axes_full, axes_ok)[0]

    f_bad, axes_bad = _solve_freq((half_ok + cv09.DX, cv09.b, cv09.d),
                                  spec_half)
    gap_bad = abs(f_full - f_bad) / f_full
    # Pozar predicts 2.702 % for a_eff = 23.876 mm; the discrete solve must
    # land near it, must have PASSED the old 5 % gate, and must FAIL the new.
    assert gap_bad == pytest.approx(0.02702, abs=0.004), f"{gap_bad:.4%}"
    assert gap_bad < 0.05, "old gate would have failed — blindness claim wrong"
    assert gap_bad > cv09.G3_TOL
    assert not cv09.geometry_gate(axes_full, axes_bad)[0]
