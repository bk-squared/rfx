"""Grading-zone assertion on the nonuniform waveguide reference-plane path.

The NU S-matrix lane shifts each port's modal waves from the plane they were
recorded on to the requested reference plane with one ``exp(-/+ j*beta*shift)``
whose beta is evaluated at the grid's BOUNDARY cell (``cfg.dx`` =
``NonUniformGrid.dx``). That is exact only when every cell between the port
plane and the reference plane has one size. ``rfx.api._sparams``
``_assert_nu_shift_span_in_one_grading_zone`` reads the cells the span
crosses from the grid's per-cell arrays and raises ``ValueError`` when more
than one size appears; a span inside one uniform zone -- coarse OR fine --
passes, and the boundary-vs-local beta difference inside a uniform fine zone
is the documented envelope
(``tests/fixtures/waveguide_nu_beta_cell_size_envelope.json``).

Helper-level cases build the port config the lane builds
(``_build_waveguide_port_config_nu``) without time stepping; one end-to-end
case drives ``compute_waveguide_s_matrix`` on a short run to prove the
assertion is wired into the lane.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.api._sparams import _assert_nu_shift_span_in_one_grading_zone, _nu_shift_span_cells
from rfx.auto_config import smooth_grading
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners.nonuniform import _build_waveguide_port_config_nu
from rfx.sources.waveguide_port import waveguide_plane_positions

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling fixture module

_A_WG = 0.02286
_B_WG = 0.01016
_F_MAX = 12e9
_FREQS = jnp.linspace(8.2e9, 12.4e9, 4)
_DX_COARSE = 1.5e-3
_DX_FINE = 0.75e-3
_N_STEPS_BUILD = 16


def _graded_profile(pre_m: float, fine_m: float, post_m: float) -> np.ndarray:
    raw = np.concatenate([
        np.full(int(round(pre_m / _DX_COARSE)), _DX_COARSE),
        np.full(int(round(fine_m / _DX_FINE)), _DX_FINE),
        np.full(int(round(post_m / _DX_COARSE)), _DX_COARSE),
    ])
    return smooth_grading(raw, max_ratio=1.3)


def _wr90_dx_graded_sim(pre_m: float, fine_m: float, post_m: float):
    """Empty WR-90 along x with a coarse | fine | coarse ``dx_profile``."""
    dx_profile = _graded_profile(pre_m, fine_m, post_m)
    domain_x = float(np.sum(dx_profile))
    sim = Simulation(
        freq_max=_F_MAX,
        domain=(domain_x, _A_WG, _B_WG),
        dx=_DX_COARSE,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
        dx_profile=dx_profile,
    )
    return sim, domain_x


def _add_port(sim, x_m, direction, ref_m, name):
    sim.add_waveguide_port(
        x_m, direction=direction, mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=10.3e9, bandwidth=0.5, reference_plane=ref_m, name=name,
    )


def _span_check(sim, entry):
    grid = sim._build_nonuniform_grid()
    cfg = _build_waveguide_port_config_nu(sim, entry, grid, jnp.asarray(_FREQS), _N_STEPS_BUILD)
    desired = float(entry.reference_plane if entry.reference_plane is not None
                    else waveguide_plane_positions(cfg)["source"])
    return grid, cfg, desired


def test_span_crossing_transition_cells_raises_positive_port():
    """Coarse block 15 mm; port at 7.5 mm, reference at 20 mm -> crosses the 1.5 -> 0.75 grading."""
    sim, _ = _wr90_dx_graded_sim(0.015, 0.040, 0.030)
    _add_port(sim, 0.0075, "+x", 0.020, "left")
    grid, cfg, desired = _span_check(sim, sim._waveguide_ports[0])
    with pytest.raises(ValueError, match=r"along x .*crosses cells of [2-9] sizes") as exc:
        _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, "left")
    msg = str(exc.value)
    assert "'left'" in msg and "boundary cell (1.5000 mm)" in msg
    assert "1.5" in msg and "0.75" in msg, msg  # both zone sizes are named
    axis, lo, hi, sizes = _nu_shift_span_cells(grid, cfg, desired)
    assert axis == "x" and lo == pytest.approx(0.0075, abs=1e-8) and hi == pytest.approx(0.020, abs=1e-8)
    assert len({round(float(s), 9) for s in sizes}) >= 3, sizes  # coarse, transition(s), fine


def test_span_crossing_transition_cells_raises_negative_port():
    """Mirror image: -x port whose span runs back into the fine block."""
    sim, domain_x = _wr90_dx_graded_sim(0.030, 0.040, 0.015)
    _add_port(sim, domain_x - 0.0075, "-x", domain_x - 0.020, "right")
    grid, cfg, desired = _span_check(sim, sim._waveguide_ports[0])
    with pytest.raises(ValueError, match=r"port 'right'.*along x"):
        _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, "right")


def test_committed_fixture_spans_lie_in_uniform_coarse_cells():
    """The NU AD fixture: both port-to-reference spans are uniform 1.5 mm cells (no raise)."""
    from tests.unit.autodiff.test_waveguide_nu_flux_ad import _wr90_nu_sim
    sim, _ = _wr90_nu_sim()
    for entry in sim._waveguide_ports:
        grid, cfg, desired = _span_check(sim, entry)
        axis, lo, hi, sizes = _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, entry.name)
        assert axis == "x"
        assert sizes.size >= 3
        np.testing.assert_allclose(sizes, _DX_COARSE, rtol=1e-9)
        assert float(cfg.dx) == pytest.approx(_DX_COARSE)


def test_span_inside_uniform_fine_block_passes_with_local_size():
    """One uniform zone is what the assertion asks for -- not the boundary size.

    Port and reference plane both inside the fine block: every crossed cell is
    0.75 mm while beta is evaluated at the 1.5 mm boundary cell. That is the
    documented arithmetic envelope, not a raise.
    """
    sim, _ = _wr90_dx_graded_sim(0.030, 0.040, 0.030)
    _add_port(sim, 0.045, "+x", 0.050, "mid")
    grid, cfg, desired = _span_check(sim, sim._waveguide_ports[0])
    axis, lo, hi, sizes = _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, "mid")
    assert sizes.size >= 3
    np.testing.assert_allclose(sizes, _DX_FINE, rtol=1e-9)
    assert float(cfg.dx) == pytest.approx(_DX_COARSE)  # boundary cell, not the local one


def test_default_reference_plane_is_checked_too():
    """reference_plane=None still shifts (record plane -> port plane); the span is checked."""
    sim, _ = _wr90_dx_graded_sim(0.015, 0.040, 0.030)
    # Port plane 1.5 mm before the grading starts; the record plane (ref_offset=3
    # cells downstream) lands inside the transition cells.
    _add_port(sim, 0.0135, "+x", None, "edge")
    grid, cfg, desired = _span_check(sim, sim._waveguide_ports[0])
    assert desired == pytest.approx(0.0135, abs=1e-8)
    with pytest.raises(ValueError, match="crosses cells of"):
        _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, "edge")


def test_z_directed_port_reads_the_dz_profile():
    """A +z port with a graded dz_profile: the check reads dz, names axis z."""
    dz_profile = _graded_profile(0.015, 0.020, 0.015)
    domain_z = float(np.sum(dz_profile))
    sim = Simulation(
        freq_max=_F_MAX,
        domain=(_A_WG, _B_WG, domain_z),
        dx=_DX_COARSE,
        boundary=BoundarySpec(
            x=Boundary(lo="pec", hi="pec"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="cpml", hi="cpml"),
        ),
        cpml_layers=8,
        dz_profile=dz_profile,
    )
    _add_port(sim, 0.0075, "+z", 0.020, "bottom")
    grid, cfg, desired = _span_check(sim, sim._waveguide_ports[0])
    with pytest.raises(ValueError, match=r"along z"):
        _assert_nu_shift_span_in_one_grading_zone(grid, cfg, desired, "bottom")


def test_compute_waveguide_s_matrix_raises_on_a_graded_span():
    """Wired into the lane: the public call fails loudly instead of applying one beta."""
    sim, domain_x = _wr90_dx_graded_sim(0.015, 0.040, 0.030)
    _add_port(sim, 0.0075, "+x", 0.020, "left")
    _add_port(sim, domain_x - 0.015, "-x", domain_x - 0.020, "right")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"port 'left'.*along x"):
            sim.compute_waveguide_s_matrix(n_steps=8, normalize=True)
