"""cv11 -- the PEC short's realized walls, and the deleted aperture trim
(#931 lattice ownership).

Two things this case could not previously see, both build-only (no FDTD
step, the whole file runs in seconds):

1. WHERE the short's electric walls land. Nothing in `11_waveguide_port_
   wr90.py` asserted it; the only thing that could have caught a
   mis-declaration was the round-trip phase gate, whose derivation allows
   +-4 cells of reference-plane uncertainty -- four times the effect it
   would need to see, so the case passed whether the short realized at
   145/146 mm or at 146 mm alone. Under the contract a body drawn
   `x_a -> x_b` on node planes stands walls at BOTH and shorts the
   interior, so the 2 mm plug realizes 145 / 146 / 147 mm and the FIRST of
   those -- the reflection plane the phase reference uses -- is at
   PEC_SHORT_X exactly.

2. That the in-script port-aperture trim is now a DOUBLE correction. The
   script's own "SEQUENCING" paragraph predicted this: the trim existed
   because the port eigenproblem spanned n_nodes columns instead of
   n_cells, and #889 fixed that in rfx. `cfg.f_cutoff` is the witness.

The geometry comes from the script's own `_build_sim`, never a copy.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CV11 = REPO_ROOT / "validation/crossval/11_waveguide_port_wr90.py"


def _load_cv11():
    spec = importlib.util.spec_from_file_location("_cv11_realized_short", CV11)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv11():
    return _load_cv11()


def test_short_realizes_walls_on_both_drawn_faces(cv11):
    sim = cv11._build_sim(cv11.FREQS_HZ, pec_short_x=cv11.PEC_SHORT_X)
    r = cv11.assert_realized_short(sim)
    assert r["planes_m"] == pytest.approx([0.145, 0.146, 0.147], abs=1e-12)
    # The reflection plane -- the one the round-trip phase reference uses.
    assert r["planes_m"][0] == pytest.approx(cv11.PEC_SHORT_X, abs=1e-12)
    # A VOLUME: it owns cells, and no sheet was declared.
    assert r["n_cells"] > 0
    assert r["n_sheets"] == 0


def test_shared_helper_agrees_with_the_case_gate(cv11):
    """The case's own `assert_realized_short` and the branch-wide helper
    must say the same thing (#931 single-owner rule)."""
    from tests._realized_geometry import assert_wall_planes

    sim = cv11._build_sim(cv11.FREQS_HZ, pec_short_x=cv11.PEC_SHORT_X)
    assert_wall_planes(sim, 0, [0.145, 0.146, 0.147], what="cv11 PEC short")


def test_short_thickness_is_an_absolute_extent(cv11):
    """The `2 * DX_M` cell-relative spelling this script flagged since
    #722/#724 is gone; the constant realizes the identical body at
    dx = 1 mm."""
    assert cv11.PEC_SHORT_T_M == pytest.approx(0.002, abs=1e-12)
    assert cv11.PEC_SHORT_T_M == pytest.approx(2 * cv11.DX_M, abs=1e-12)
    # The expression is gone from the CODE (it survives only inside the
    # comment that records why it was replaced).
    code = [ln for ln in CV11.read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    assert not any("pec_short_x + 2 * DX_M" in ln for ln in code)


def test_the_short_declaration_is_falsifiable(cv11):
    """Move the short one cell and the gate must refuse it. Without this
    arm the assertion above could be passing for the wrong reason."""
    sim = cv11._build_sim(cv11.FREQS_HZ,
                          pec_short_x=cv11.PEC_SHORT_X + cv11.DX_M)
    with pytest.raises(RuntimeError) as exc:
        cv11.assert_realized_short(sim)
    assert "realized wall planes" in str(exc.value)


def test_the_front_wall_is_the_whole_cross_section(cv11):
    """The plug reaches the REALIZED guide walls (23 x 11 mm), so every
    tangential edge on its front plane is PEC — 276 Ey and 264 Ez on the
    24 x 12 node plane. This is the check that was missing on 2026-09-06:
    the plug was drawn to the declared 22.86 x 10.16 mm, its top face
    rounded to z = 10.000 mm under a wall at 11.000 mm, and the one-cell
    slot read max||S11|-1| = 0.0560 on VESSL 369367259194 (adjudicated
    2026-09-07, scripts/diagnostics/pec_short_lane_ab.py)."""
    sim = cv11._build_sim(cv11.FREQS_HZ, pec_short_x=cv11.PEC_SHORT_X)
    r = cv11.assert_realized_short(sim)
    assert r["front_wall_full"]
    assert r["front_wall"] == {"ey_pec": 276, "ey_full": 276,
                               "ez_pec": 264, "ez_full": 264}
    assert r["n_cells"] == 2 * 23 * 11


def test_a_plug_drawn_to_the_declared_cross_section_is_refused(cv11):
    """The falsifier arm of the test above: the 2026-09-06 drawing (hi
    corner at DOMAIN_Y x DOMAIN_Z) realizes 240/264 Ez and 253/276 Ey on
    the front plane and the gate must refuse it by name."""
    sim = cv11._build_sim(cv11.FREQS_HZ, pec_short_x=None)
    sim.add(cv11.Box((cv11.PEC_SHORT_X, 0.0, 0.0),
                     (cv11.PEC_SHORT_X + cv11.PEC_SHORT_T_M,
                      cv11.DOMAIN_Y, cv11.DOMAIN_Z)),
            material="pec")
    with pytest.raises(RuntimeError) as exc:
        cv11.assert_realized_short(sim)
    msg = str(exc.value)
    assert "not the full cross-section" in msg
    assert "Ez 240/264" in msg and "Ey 253/276" in msg


def test_port_cutoff_no_longer_needs_the_in_script_aperture_trim(cv11):
    """#889 fixed the rfx-side default (`_node_span_to_cell_span`), so the
    untrimmed port now solves the guide's 23 cells and its cutoff lands on
    the quote-realized reference. This is the measurement that deleted the
    trim, kept as a test so the trim cannot come back silently."""
    import jax.numpy as jnp

    sim = cv11._build_sim(cv11.FREQS_HZ)
    grid = sim._build_grid()
    cfg = sim._build_waveguide_port_config(
        sim._waveguide_ports[0], grid, jnp.asarray(cv11.FREQS_HZ), 100)
    f_cutoff = float(cfg.f_cutoff)
    # -0.080% against the quote-realized 6.517391 GHz. With the trim this
    # read 6.807677 GHz (+4.454%) -- one cell too narrow a guide.
    assert f_cutoff == pytest.approx(6.512162e9, rel=1e-6)
    rel = (f_cutoff - cv11.F_CUTOFF_TE10) / cv11.F_CUTOFF_TE10
    assert abs(rel) < 0.002
    code = [ln for ln in CV11.read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    assert not any("aperture_kw" in ln for ln in code)


def test_the_dielectric_slab_realization_is_untouched(cv11):
    """#931 §1.8 fences dielectric sampling (node, half-open). cv11's slab
    must realize exactly what it realized before -- 11 occupied x-nodes,
    95.000 to 105.000 mm -- or the contract touched something it promised
    not to."""
    import numpy as np
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    centre = 0.5 * (cv11.PORT_LEFT_X + cv11.PORT_RIGHT_X)
    lo = (centre - 0.005, 0.0, 0.0)
    hi = (centre + 0.005, cv11.DOMAIN_Y, cv11.DOMAIN_Z)
    sim = cv11._build_sim(cv11.FREQS_HZ, obstacles=[(lo, hi, 2.0)])
    grid = sim._build_grid()
    mats = sim._assemble_materials(grid)[0]
    eps = np.asarray(mats.eps_r)[:, grid.shape[1] // 2, grid.shape[2] // 2]
    occ = np.flatnonzero(np.isclose(eps, 2.0))
    xs = np.asarray(coords_from_uniform_grid(grid).x)
    assert occ.size == 11
    assert xs[occ.min()] == pytest.approx(0.095, abs=1e-9)
    assert xs[occ.max()] == pytest.approx(0.105, abs=1e-9)
