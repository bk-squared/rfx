"""Tests for the public ``port_reference_sims`` plumbing on
``compute_waveguide_s_matrix`` — per-port matched-straight-guide references
for interior-PEC multi-port junctions (T-junctions / branches / septa).

Coverage
--------
* Guard raises (milliseconds — they fire BEFORE any FDTD): normalize must be
  ``'flux'``; one reference Simulation per waveguide port; each reference grid
  must match the device grid (shape + dx); multimode and eps/sigma-override
  combinations are unsupported.
* Advisory warning: a compact T-junction whose probe planes sit on top of the
  junction must fire the clearance advisory (band kept below the TE20 cutoff so
  the advisory is not skipped).
* Behavioral witness: on the compact T-junction the matched reference lowers
  ``|S11|`` (vacuum-reference blow-up 11.0 -> 3.15 here; |S11| band-mean well
  below 1) yet the overall matrix stays NON-physical (max|S| > 1.05), locking
  the necessary-but-not-sufficient finding. Those two numbers were 9.8 -> 3.1
  before the port aperture was corrected to the guide's N cells: this port
  declares ``y_range=(0.04, 0.08)`` inside a 0.12 m domain with interior PEC-box
  walls, so its transverse span is a SUB-APERTURE and the +face DROP weight
  never fired on ``u``. The pre-correction template therefore carried a
  full-weight cell one past the wall (21 x 11 cells, ``aperture_dA.sum()``
  8.4000e-04, ``f_cutoff`` 3.565631 GHz); corrected it is 20 x 10, 8.0000e-04
  and 3.743554 GHz. The assertions here are inequalities and hold either way;
  the two figures above are prose and are refreshed with the correction. Companion committed evidence lives
  at ``tests/fixtures/waveguide_tjunction_e4/`` /
  ``tests/crossval/test_waveguide_tjunction_e4e5_gates.py``.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box
from tests._realized_geometry import assert_wall_planes, realized


# --------------------------------------------------------------------------
# Tiny two-port builders — used only by the guard tests, which raise before
# any FDTD, so the run parameters below are never exercised.
# --------------------------------------------------------------------------

def _tiny_two_port(*, dx=0.004, n_modes=1):
    s = Simulation(
        freq_max=10e9, domain=(0.12, 0.04, 0.02),
        boundary="cpml", cpml_layers=10, dx=dx,
    )
    common = dict(
        mode=(1, 0), mode_type="TE", freqs=jnp.linspace(4.5e9, 8e9, 3),
        f0=6e9, ref_offset=3, probe_offset=8, n_modes=n_modes,
    )
    s.add_waveguide_port(0.01, direction="+x", name="a", **common)
    s.add_waveguide_port(0.11, direction="-x", name="b", **common)
    return s


@pytest.mark.parametrize("normalize", [False, True])
def test_port_reference_sims_requires_flux(normalize):
    s = _tiny_two_port()
    refs = [_tiny_two_port(), _tiny_two_port()]
    with pytest.raises(ValueError, match="requires normalize='flux'"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize=normalize, port_reference_sims=refs,
        )


def test_port_reference_sims_wrong_length_raises():
    s = _tiny_two_port()
    with pytest.raises(ValueError, match="one Simulation per waveguide port"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux", port_reference_sims=[_tiny_two_port()],
        )


def test_port_reference_sims_mismatched_grid_raises():
    s = _tiny_two_port(dx=0.004)
    bad = _tiny_two_port(dx=0.002)  # different dx -> different grid shape
    with pytest.raises(ValueError, match="must match the device grid"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux",
            port_reference_sims=[bad, _tiny_two_port()],
        )


def test_port_reference_sims_multimode_raises():
    s = _tiny_two_port(n_modes=2)
    refs = [_tiny_two_port(), _tiny_two_port()]
    with pytest.raises(NotImplementedError, match="multimode"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux", port_reference_sims=refs,
        )


def test_port_reference_sims_eps_override_combo_raises():
    s = _tiny_two_port()
    grid = s._build_grid()
    with pytest.raises(NotImplementedError, match="eps_override"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux",
            eps_override=jnp.ones(grid.shape),
            port_reference_sims=[_tiny_two_port(), _tiny_two_port()],
        )


# --------------------------------------------------------------------------
# Compact T-junction builders (geometry copied from the SKIPPED
# test_api.py::test_waveguide_branch_junction_mixed_normals_reciprocal_through_api).
# The main guide runs horizontally (y in [0.04, 0.08]); the top arm opens at
# x in [0.04, 0.08], y in [0.08, 0.12].
#
# The walls are PEC VOLUMES (#931 §1.2). Under the ownership contract each
# block realizes tangential walls on BOTH of its drawn faces, so the guide
# between the y = 0.04 and y = 0.08 faces is 40.0 mm — the number the
# declaration states. Before the contract a body's far face was never a
# wall, the metal ended one node short on each side and the same geometry
# realized a 42.0 mm guide; the ports' declared ``y_range=(0.04, 0.08)``
# lands ON the walls for the first time. Anything this file's prose quotes
# that was measured on the 42 mm guide (aperture areas, cutoffs, the
# |S11| blow-up figures) is a pre-#931 measurement and is re-read with the
# fixtures, not translated.
# ``test_tj_walls_are_realized_where_they_are_drawn`` below is the
# build-time witness, and it costs no solve.
# --------------------------------------------------------------------------

def _tj_common(freqs, f0):
    return dict(
        mode=(1, 0), mode_type="TE", freqs=freqs, f0=f0,
        ref_offset=3, probe_offset=15, z_range=(0.00, 0.02),
    )


def _tj_sim():
    return Simulation(
        freq_max=10e9, domain=(0.12, 0.12, 0.02),
        boundary="cpml", cpml_layers=10, dx=0.002,
    )


def _tj_device(freqs, f0):
    s = _tj_sim()
    s.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    s.add(Box((0.0, 0.08, 0.0), (0.04, 0.12, 0.02)), material="pec")
    s.add(Box((0.08, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = _tj_common(freqs, f0)
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return s


def _tj_ref_horizontal(freqs, f0):
    """Straight horizontal guide (walls y in [0,0.04] and [0.08,0.12], full x)
    — the matched continuation for the left and right ports (no top arm)."""
    s = _tj_sim()
    s.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    s.add(Box((0.0, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = _tj_common(freqs, f0)
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return s


def _tj_ref_vertical(freqs, f0):
    """Straight vertical guide (walls x in [0,0.04] and [0.08,0.12], full y)
    — the matched continuation for the top port (no horizontal arm)."""
    s = _tj_sim()
    s.add(Box((0.0, 0.0, 0.0), (0.04, 0.12, 0.02)), material="pec")
    s.add(Box((0.08, 0.0, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = _tj_common(freqs, f0)
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return s


def _tj_refs(freqs, f0):
    return [
        _tj_ref_horizontal(freqs, f0),   # left  (+x)
        _tj_ref_horizontal(freqs, f0),   # right (-x)
        _tj_ref_vertical(freqs, f0),     # top   (-y)
    ]


def test_port_reference_sims_clearance_advisory_fires(monkeypatch):
    """Probe planes sitting on top of the junction must fire the clearance
    advisory. The band is kept below the TE20 cutoff (fc2 = C0/a = 7.5 GHz for
    a = 0.04 m) so the advisory is not skipped for an in-band higher mode."""
    freqs = jnp.linspace(4.5e9, 6.5e9, 3)
    f0 = 5.5e9

    class ReachedExtractor(Exception):
        pass

    def stop_before_solve(*args, **kwargs):
        raise ReachedExtractor

    monkeypatch.setattr(
        # #980 Phase 2: compute_waveguide_s_matrix's body lives in
        # rfx/sparams/waveguide.py, so this is where the extractor name is
        # looked up. The re-export in rfx.api._sparams would still accept the
        # patch and silently not be the binding the lane reads.
        "rfx.sparams.waveguide.extract_waveguide_s_matrix_flux",
        stop_before_solve,
    )
    with pytest.warns(UserWarning) as record, pytest.raises(ReachedExtractor):
        _tj_device(freqs, f0).compute_waveguide_s_matrix(
            num_periods=8, normalize="flux", port_reference_sims=_tj_refs(freqs, f0),
        )
    findings = [w.message for w in record
                if getattr(w.message, "code", None) == "port_junction_probe_clearance"]
    assert [finding.loc for finding in findings] == ["port:0", "port:1", "port:2"]
    assert all(finding.severity == "warning" for finding in findings)


@pytest.mark.parametrize("subpixel_smoothing", [None, "kottke_pec"])
def test_identical_port_references_have_no_junction_clearance_advisory(
    monkeypatch, subpixel_smoothing,
):
    """Kottke clearing solver edge masks must not erase advisory geometry."""
    freqs = jnp.linspace(4.5e9, 6.5e9, 3)

    class ReachedExtractor(Exception):
        pass

    def stop_before_solve(*args, **kwargs):
        # Preserve the solver's dispatch: Kottke owns its inverse-eps
        # tensor and must not acquire an additional staircase PEC mask.
        assert (kwargs["pec_edge_masks"] is None) == (
            subpixel_smoothing == "kottke_pec"
        )
        assert all(edges is not None
                   for edges in kwargs["ref_pec_edge_masks_per_port"])
        raise ReachedExtractor

    monkeypatch.setattr(
        # #980 Phase 2: compute_waveguide_s_matrix's body lives in
        # rfx/sparams/waveguide.py, so this is where the extractor name is
        # looked up. The re-export in rfx.api._sparams would still accept the
        # patch and silently not be the binding the lane reads.
        "rfx.sparams.waveguide.extract_waveguide_s_matrix_flux",
        stop_before_solve,
    )
    with warnings.catch_warnings(record=True) as record, pytest.raises(ReachedExtractor):
        warnings.simplefilter("always")
        _tj_ref_horizontal(freqs, 5.5e9).compute_waveguide_s_matrix(
            num_periods=8, normalize="flux", subpixel_smoothing=subpixel_smoothing,
            port_reference_sims=[_tj_ref_horizontal(freqs, 5.5e9) for _ in range(3)],
        )
    assert not any(getattr(w.message, "code", None) == "port_junction_probe_clearance"
                   for w in record)


@pytest.mark.parametrize("difference,expected", [
    ("none", False), ("near_sheet", True), ("far_sheet", False),
    ("edge_component", True), ("sigma", True),
    ("device_only", True), ("reference_only", True),
])
def test_junction_clearance_reads_material_and_component_edges(difference, expected):
    """PEC sheets carry no sigma; identical or distant guides stay silent.

    Component changes must remain visible even if the union of PEC edge
    locations is unchanged (different conductor orientations).
    """
    from rfx.api._sparams import _warn_junction_probe_clearance

    shape = (8, 2, 2)
    dev_sigma = np.zeros(shape)
    ref_sigma = np.zeros(shape)
    dev_edges = [np.zeros(shape, dtype=bool) for _ in range(3)]
    ref_edges = [np.zeros(shape, dtype=bool) for _ in range(3)]
    if difference in ("near_sheet", "far_sheet", "edge_component"):
        plane = 7 if difference == "far_sheet" else 3
        dev_edges[1][plane] = True
    if difference == "edge_component":
        ref_edges[2][3] = True
        np.testing.assert_array_equal(
            np.logical_or.reduce(dev_edges), np.logical_or.reduce(ref_edges),
        )
    if difference == "sigma":
        dev_sigma[3] = 1.0
    if difference == "device_only":
        dev_edges[1][3] = True
        ref_edges = None
    if difference == "reference_only":
        ref_edges[1][3] = True
        dev_edges = None
    cfg = SimpleNamespace(a=0.04, normal_axis="x", probe_x=3)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_junction_probe_clearance(
            SimpleNamespace(dx=0.02), [cfg], dev_sigma, [ref_sigma],
            np.array([4.5e9, 6.5e9]),
            device_pec_edges=dev_edges, ref_pec_edges=[ref_edges],
        )
    findings = [w.message for w in record]
    assert [finding.code for finding in findings] == (
        ["port_junction_probe_clearance"] if expected else []
    )


def test_port_reference_sims_compact_junction_necessary_not_sufficient():
    """A/B witness on the compact T-junction: the matched reference lowers
    |S11| (vacuum blow-up -> physical diagonal) but the overall matrix stays
    non-physical (max|S| > 1.05) and the passivity self-check still fires —
    locking the necessary-but-not-sufficient finding in BOTH directions.

    Full band (includes the TE20-propagating region) so the vacuum reference
    blows up the way the 2026-07-06 verification documented (max|S| ~ 9.8)."""
    freqs = jnp.linspace(4.5e9, 8.0e9, 6)
    f0 = 6e9
    num_periods = 30

    s_vac = np.asarray(
        _tj_device(freqs, f0)
        .compute_waveguide_s_matrix(num_periods=num_periods, normalize="flux")
        .s_params
    )
    # The extractor's passivity self-check MUST still fire with references —
    # the compact-geometry S-matrix is still non-physical (locked).
    with pytest.warns(UserWarning, match="passiv"):
        result = _tj_device(freqs, f0).compute_waveguide_s_matrix(
            num_periods=num_periods, normalize="flux",
            port_reference_sims=_tj_refs(freqs, f0),
        )
    s_ref = np.asarray(result.s_params)

    s11_vac = float(np.mean(np.abs(s_vac[0, 0, :])))
    s11_ref = float(np.mean(np.abs(s_ref[0, 0, :])))
    max_vac = float(np.max(np.abs(s_vac)))
    max_ref = float(np.max(np.abs(s_ref)))

    # Direction 1 — the matched reference fixes the reflection diagonal.
    assert s11_ref < s11_vac            # lower than the vacuum reference
    assert s11_ref < 1.0                # physical reflection band-mean
    # Direction 2 — the overall matrix is still non-physical (compact geometry).
    assert max_ref > 1.05               # non-passive residual remains
    assert max_ref < max_vac            # but the blow-up is reduced


def test_tj_walls_are_realized_where_they_are_drawn():
    """Build-time witness (no solve) for the T-junction's guide width.

    The two horizontal wall blocks are drawn to y = 0.04 m and from
    y = 0.08 m at dx = 2 mm, both on node lines. A volume owns the cells
    its centres fall in and walls both drawn faces, so the realized guide
    runs from node 0.04 m to node 0.08 m — 20 cells, 40.0 mm, the drawn
    gap exactly. This is the same geometry
    ``tests/unit/ports/test_port_aperture_rasterization.py`` measures
    through preflight, asserted here at the source.
    """
    freqs = jnp.linspace(4.5e9, 6.5e9, 3)
    sim = _tj_ref_horizontal(freqs, 5.5e9)
    rz = realized(sim)
    pad = rz.grid.axis_pads[1]
    expected = list(range(pad, pad + 21)) + list(range(pad + 40, pad + 61))
    assert_wall_planes(sim, 1, expected_planes=expected,
                       what="T-junction guide walls")
    inner_lo, inner_hi = pad + 20, pad + 40
    assert (inner_hi - inner_lo) * rz.grid.dx == pytest.approx(0.040, rel=1e-12)
