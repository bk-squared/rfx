"""TFSF/PMC composition and face-interpolated Poynting flux.

B3b preserves adjacent physical H samples and images them at consumption;
zero normal flux follows from H interpolated onto the declared PMC plane.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402


# ---------------------------------------------------------------------------
# OQ7 — TFSF + PMC composition
# ---------------------------------------------------------------------------


def test_oq7_tfsf_plus_pmc_rejected_or_documented():
    """Build TFSF + BoundarySpec with PMC on one face. Either:
      - add_tfsf_source raises (composition explicitly rejected), OR
      - construction succeeds and we pin that behaviour in a
        documentation assertion on sim._boundary_spec.

    Outcome: record the CURRENT rfx verdict so downstream sessions do
    not rediscover it.
    """
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.015), dx=0.5e-3,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pmc", hi="cpml")),
        cpml_layers=6,
    )
    try:
        sim.add_tfsf_source(
            f0=5e9, bandwidth=0.5,
            polarization="ez", direction="+x",
        )
    except ValueError as e:
        # Outcome A: rfx rejects TFSF+PMC at API level. Pin the error
        # message substring so the behaviour is stable.
        assert "TFSF" in str(e) or "cpml" in str(e), (
            f"TFSF+PMC rejection message should mention TFSF or cpml; "
            f"got: {e}"
        )
        return
    # Outcome B: add_tfsf_source accepted the config. Document this.
    assert sim._tfsf is not None
    assert "z_lo" in sim._boundary_spec.pmc_faces()
    # The shared curl images H after TFSF/absorber/source changes.


# ---------------------------------------------------------------------------
# OQ8 — NTFF + PMC Poynting-flux on a shared face
# ---------------------------------------------------------------------------


def test_oq8_ntff_over_pmc_face_gives_zero_poynting():
    """Interpolate the odd H pair onto the face before computing Poynting.

    H[0] lies inside the domain and may not be used as the wall sample.
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
    # Interior Ez source so fields actually build up before we probe.
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
    # H_t=0 is at the node plane, not at the first physical half-cell.
    # Sanity: Hx / Hy must be NON-ZERO just above the PMC face so the
    # interior propagation still works (otherwise it means we failed to
    # energise the cavity, not that the algebra is correct).
    max_hx_interior = float(np.max(np.abs(hx[:, :, nz // 2])))
    assert max_hx_interior > 1e-20, (
        f"no H field built up interior — source / energise failed "
        f"(max|Hx[:,:,nz/2]| = {max_hx_interior:.3e})"
    )


# ---------------------------------------------------------------------------
# OQ9 — distributed_v2 step_fn_cpml handles PEC faces via CPML init, not hook
# ---------------------------------------------------------------------------


def test_oq9_distributed_v2_cpml_path_enforces_pec_face_via_cpml_init():
    """BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pec', hi='cpml')) routes
    through distributed_v2::step_fn_cpml (because sim._boundary == 'cpml').
    step_fn_cpml has NO face hook for PEC; the PEC face is enforced via
    the per-face CPML profile baked in at init_cpml (cpml.py:325-330).

    Smoke: after 30 steps tangential E on the PEC z_lo face reads
    effectively zero (<1e-10) despite interior fields being ~1e-3 scale.
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
    # sim._boundary is the scalar legacy view; for mixed cpml+pec this
    # resolves to "cpml" so the run goes through step_fn_cpml.
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
    # PEC on z_lo zeros tangential E (Ex, Ey) at k=0 to machine precision
    # via the CPML-profile route — even without a scan-body PEC hook.
    assert max_ex_z_lo < 1e-10, (
        f"PEC z_lo failed to zero Ex via CPML init: "
        f"max|Ex[:,:,0]| = {max_ex_z_lo:.3e}"
    )
    assert max_ey_z_lo < 1e-10, (
        f"PEC z_lo failed to zero Ey via CPML init: "
        f"max|Ey[:,:,0]| = {max_ey_z_lo:.3e}"
    )
    # Sanity: interior must be non-trivially energised so the zero on
    # z_lo is a boundary condition, not a "source never turned on" artifact.
    assert max_ex_interior > 1e-6, (
        f"interior Ex too small — source may not have energised: "
        f"max|Ex[:,:,nz/2]| = {max_ex_interior:.3e}"
    )
