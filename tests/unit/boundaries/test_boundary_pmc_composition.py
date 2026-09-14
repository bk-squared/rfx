"""OQ7 & OQ8 — TFSF+PMC and NTFF+PMC composition surfaces.

These are documentation-tests: they pin the CURRENT behaviour of
mixed-boundary setups rather than asserting a "correct" answer. When a
setup turns out to be ambiguous or dangerous, the test fails loudly so
future sessions pick up the tracking item.

Per the T8/2026-04 open-questions log (.omc/plans/open-questions.md §OQ7/8),
these are T10 candidates tracked for visibility, not blockers.

OQ7 — TFSF + PMC scan-body ordering:
  simulation.py:696-710 applies H in order:
    (1) update_h
    (2) apply_tfsf_h      [injects tangential H on TFSF box face]
    (3) apply_cpml_h
    (4) apply_pmc_faces   [zeroes tangential H on PMC face]
  If the TFSF box face coincides with a PMC face, the PMC zero fires
  after the TFSF inject, silently killing the injected wave on that
  cell. The hypothesis is that rfx either (a) rejects this composition
  at add_tfsf_source time, or (b) accepts it and zeroes the overlap.

OQ8 — NTFF + PMC Poynting-flux:
  For a Huygens-box face that coincides with a PMC face, tangential H
  is zero by the PMC condition. The Poynting flux normal to that face
  is ``P_n = Re(E_tan x H_tan*)``. If ``H_tan = 0``, then ``P_n = 0``.
  The NTFF surface integral over a PMC face therefore contributes
  identically zero — no "silent meaningless flux" error. This test
  pins that algebraic fact at runtime.

OQ9 — `step_fn_cpml` missing PEC face hook:
  Pre-existing architecture: when ``sim._boundary == "cpml"`` and
  pec_faces is non-empty, the CPML init in ``rfx/boundaries/cpml.py``
  bakes PEC into the per-face CPML profile via ``_lo_face_profile``
  / ``_hi_face_profile`` — the CPML machinery enforces PEC on those
  faces automatically, no scan-body hook required.

  Round 4 of the B0 distributed-admission review (2026-09-15) established
  by measurement that this holds on the SINGLE-DEVICE lane only. The
  distributed runners have no per-face clamp on ANY axis:
  ``_init_cpml_distributed`` builds one scalar ``_cpml_profile`` and the
  kernel drives it at y-lo/y-hi/z-lo/z-hi unconditionally, so this very
  fixture was 90.58 % wrong at the source probe on ``devices=devices[:2]``
  with 0 warnings — and stayed green here because the assertions are a
  zero-PATTERN on the PEC face and never a value. The structural claim is
  therefore pinned on the uniform lane, where the mechanism exists, and
  the distributed lane's obligation (a named refusal, B0 class 6) is
  pinned separately. Both tests are below, with the RED numbers.
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
    # The scan body WILL zero TFSF H on z_lo (per simulation.py:696-710
    # ordering). This is documented behaviour, not a bug — users must
    # keep the TFSF box inside PMC faces by >= 1 cell.


# ---------------------------------------------------------------------------
# OQ8 — NTFF + PMC Poynting-flux on a shared face
# ---------------------------------------------------------------------------


def test_oq8_ntff_over_pmc_face_gives_zero_poynting():
    """Directly verify the algebraic claim: on a PMC face, tangential H
    is exactly zero after apply_pmc_faces, so any Poynting flux summed
    over that face is also zero.

    Build a minimal sim with z_lo=PMC. Inject an interior ez source.
    Run a few steps. Assert H-tangential on z_lo face cells is zero
    (bit-precise — apply_pmc_faces writes literal 0.0)."""
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
    hy = np.asarray(st.hy)
    # PMC zeroes Hx and Hy on z_lo (tangential H components for z-face).
    max_hx_z_lo = float(np.max(np.abs(hx[:, :, 0])))
    max_hy_z_lo = float(np.max(np.abs(hy[:, :, 0])))
    assert max_hx_z_lo == 0.0, (
        f"PMC z_lo failed to zero Hx: max|Hx[:,:,0]| = {max_hx_z_lo:.3e}"
    )
    assert max_hy_z_lo == 0.0, (
        f"PMC z_lo failed to zero Hy: max|Hy[:,:,0]| = {max_hy_z_lo:.3e}"
    )
    # Sanity: Hx / Hy must be NON-ZERO just above the PMC face so the
    # interior propagation still works (otherwise it means we failed to
    # energise the cavity, not that the algebra is correct).
    max_hx_interior = float(np.max(np.abs(hx[:, :, nz // 2])))
    assert max_hx_interior > 1e-20, (
        f"no H field built up interior — source / energise failed "
        f"(max|Hx[:,:,nz/2]| = {max_hx_interior:.3e})"
    )


# ---------------------------------------------------------------------------
# OQ9 — a PEC face composed with CPML: handled on the uniform lane by the
#       per-face CPML init, REFUSED on the distributed lane (B0 class 6)
# ---------------------------------------------------------------------------


def _oq9_sim():
    """The OQ9 fixture, unchanged from the original test.

    ``BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pec', hi='cpml'))``
    resolves ``sim._boundary`` to the scalar ``'cpml'``, so it routes
    through the CPML step body on whichever lane runs it.
    """
    dx = 5e-3
    nx, ny, nz = 16, 8, 24
    sim = Simulation(
        freq_max=5e9, domain=(nx * dx, ny * dx, nz * dx), dx=dx,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=6,
    )
    sim.add_source((nx // 2 * dx, ny // 2 * dx, nz // 2 * dx), "ex")
    sim.add_probe(((nx // 2 + 1) * dx, ny // 2 * dx, nz // 2 * dx), "ex")
    return sim, nz


def test_oq9_uniform_cpml_path_enforces_pec_face_via_cpml_init():
    """The OQ9 structural claim, on the lane that actually implements it.

    ``step_fn_cpml`` has NO face hook for PEC; on the SINGLE-DEVICE lane the
    PEC face is enforced via the per-face CPML profile baked in at
    ``init_cpml`` (``rfx/boundaries/cpml.py``, which clamps each face's
    profile to that face's allocated pad -- 0 on a PEC face).

    Smoke: after 30 steps tangential E on the PEC z_lo face reads
    effectively zero (<1e-10) despite interior fields being ~1e-3 scale.

    Round 4 of the B0 review moved this test from ``devices=devices[:2]`` to
    the single-device lane, because the DISTRIBUTED lane does not implement
    the per-face clamp at all and this fixture is 90.58 % wrong there -- see
    ``test_oq9_distributed_v2_refuses_the_pec_face_composition`` below. The
    assertions are unchanged; only the lane they are made on is, because it
    is the lane whose mechanism the docstring describes.
    """
    sim, nz = _oq9_sim()
    # sim._boundary is the scalar legacy view; for mixed cpml+pec this
    # resolves to "cpml" so the run goes through step_fn_cpml.
    assert sim._boundary == "cpml"
    assert "z_lo" in sim._boundary_spec.pec_faces()
    result = sim.run(n_steps=30, compute_s_params=False)
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


def test_oq9_distributed_v2_refuses_the_pec_face_composition():
    """On the DISTRIBUTED lane this composition is refused, and must be.

    The mechanism the test above pins does not exist on the distributed
    lane: ``_init_cpml_distributed`` (``rfx/runners/distributed.py``) builds
    ONE scalar ``_cpml_profile(grid.cpml_layers, ...)`` and the kernel
    applies it at y-lo/y-hi/z-lo/z-hi unconditionally, never reading
    ``grid.face_pads``.  So a ``pec``/``pmc`` y or z face gets absorbed at.

    RED, measured on ``origin/main`` 883615c6 (identical digits on the B0
    branch before it was widened, and 0 warnings every time), 24x8x8 mm at
    dx=1 mm, ``cpml_layers=8``, ``amplitude_kind='field'`` Ez source at
    (6, 4, 4) mm, Ez probes at x = 6/12/20 mm, 60 steps, 2 virtual CPU
    devices -- i.e. THIS test's boundary composition::

        source probe x=6 mm    max|dEz| 4.003862e+00 on 4.420338e+00
                               = 90.5782 % of its own peak
        probe x=12 mm          409.4824 % of its own peak
        probe x=20 mm          281.3395 % of its own peak
        v1 pmap at 1 device    bit-for-bit the same three figures
        symmetric control      1.078195e-07 of peak

    This test used to run that configuration at ``devices=devices[:2]`` and
    stay green, because it asserts a zero-PATTERN on the PEC face and never
    a value -- the same false green the x face had in
    ``test_boundary_pmc_distributed.py``.  B0 class 6 now refuses it
    (``rfx.runners.distributed_v2.check_absorber_faces_are_absorbing``), so
    what the distributed lane owes this fixture is a named refusal, and that
    is what is asserted here.
    """
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("need 2 virtual devices for distributed_v2 routing")
    sim, _nz = _oq9_sim()
    with pytest.raises(ValueError) as excinfo:
        sim.run(n_steps=30, devices=devices[:2], compute_s_params=False)
    msg = str(excinfo.value)
    assert "faces z-lo" in msg or "face z-lo" in msg, msg
    assert "no CPML absorber" in msg, msg
    # the refusal must say how to proceed, and the only honest route for a
    # model that wants a reflector face is the single-device lane.
    assert "omit devices=" in msg, msg
