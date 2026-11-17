"""TFSF/PMC composition and face-interpolated Poynting flux.

B3b preserves adjacent physical H samples and images them at consumption;
zero normal flux follows from H interpolated onto the declared PMC plane.
"""

from __future__ import annotations

import jax
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402




def test_oq7_tfsf_plus_pmc_rejected_or_documented():
    """Explicit PMC/CPML side faces cannot be replaced by periodic faces."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.015), dx=0.5e-3,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pmc", hi="cpml")),
        cpml_layers=6,
    )
    with pytest.raises(ValueError, match="y_lo='cpml'.*z_lo='pmc'.*z_hi='cpml'"):
        sim.add_tfsf_source(
            f0=5e9, bandwidth=0.5,
            polarization="ez", direction="+x",
        )




def test_oq8_ntff_over_pmc_face_gives_zero_poynting():
    """Interpolate the odd H pair onto the face before computing Poynting.

    H[0] lies inside the domain and may not be used as the wall sample.

    Hx / Hy must be non-zero just above the PMC face so the interior propagation still works; otherwise the cavity was not energised.
    """
    dx = 0.5e-3
    nx, ny, nz = 16, 16, 16
    sim = Simulation(
        freq_max=10e9,
        domain=(nx * dx, ny * dx, nz * dx), dx=dx,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pmc", hi="cpml")),
        cpml_layers=6,
    )
    sim.add_source(((nx // 2) * dx, (ny // 2) * dx, (nz // 2) * dx), "ez")
    sim.add_probe(((nx // 2 + 1) * dx, (ny // 2) * dx, (nz // 2) * dx), "ez")
    res = sim.run(n_steps=30, compute_s_params=False)
    st = res.state
    hx = np.asarray(st.hx)
    from rfx.core.yee import CurlBoundary, h_neighbor
    boundary = CurlBoundary(pmc_faces=frozenset({"z_lo"}))
    for field in (st.hx, st.hy):
        face_h = (field + h_neighbor(field, 2, boundary=boundary))[:, :, 0] / 2
        np.testing.assert_array_equal(np.asarray(face_h), 0.)
    max_hx_interior = float(np.max(np.abs(hx[:, :, nz // 2])))
    assert max_hx_interior > 1e-20, (
        f"no H field built up interior — source / energise failed "
        f"(max|Hx[:,:,nz/2]| = {max_hx_interior:.3e})"
    )




def test_oq9_distributed_v2_cpml_path_enforces_pec_face_via_cpml_init():
    """BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pec', hi='cpml')) routes
    through distributed_v2::step_fn_cpml (because sim._boundary == 'cpml').
    step_fn_cpml has NO face hook for PEC; the PEC face is enforced via
    the per-face CPML profile baked in at init_cpml (cpml.py:325-330).

    Smoke: after 30 steps tangential E on the PEC z_lo face reads
    effectively zero (<1e-10) despite interior fields being ~1e-3 scale.

    Interior must be non-trivially energised so the zero on z_lo is a boundary condition, not a "source never turned on" artifact.
    """
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("need 2 virtual devices for distributed_v2 routing")
    dx = 5e-3
    nx, ny, nz = 16, 8, 24
    sim = Simulation(
        freq_max=5e9, domain=(nx * dx, ny * dx, nz * dx), dx=dx,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=6,
    )
    assert sim._boundary == "cpml"
    assert "z_lo" in sim._boundary_spec.pec_faces()
    sim.add_source((nx // 2 * dx, ny // 2 * dx, nz // 2 * dx), "ex")
    sim.add_probe(((nx // 2 + 1) * dx, ny // 2 * dx, nz // 2 * dx), "ex")
    result = sim.run(n_steps=30, devices=devices[:2], compute_s_params=False)
    ex = np.asarray(result.state.ex)
    ey = np.asarray(result.state.ey)
    max_ex_z_lo = float(np.max(np.abs(ex[:, :, 0])))
    max_ey_z_lo = float(np.max(np.abs(ey[:, :, 0])))
    max_ex_interior = float(np.max(np.abs(ex[:, :, nz // 2])))
    assert max_ex_z_lo < 1e-10, (
        f"PEC z_lo failed to zero Ex via CPML init: "
        f"max|Ex[:,:,0]| = {max_ex_z_lo:.3e}"
    )
    assert max_ey_z_lo < 1e-10, (
        f"PEC z_lo failed to zero Ey via CPML init: "
        f"max|Ey[:,:,0]| = {max_ey_z_lo:.3e}"
    )
    assert max_ex_interior > 1e-6, (
        f"interior Ex too small — source may not have energised: "
        f"max|Ex[:,:,nz/2]| = {max_ex_interior:.3e}"
    )
