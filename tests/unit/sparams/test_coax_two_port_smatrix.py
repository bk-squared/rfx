"""``compute_coaxial_two_port`` — the two-drive coax S-matrix (issue #489 stage 2).

One file for the S-matrix side of the coax lane (tier 3b of the 2026-09
test-corpus reorganisation, see
``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``). Sections,
each formerly its own file:

1. ``Simulation.compute_coaxial_two_port`` FDTD path — was
   ``test_coax_two_port_fdtd.py``. STATUS: VALIDATED WITH SCOPE (issue #489,
   See the docstring on ``CoaxialTwoPortResult`` for the full
   evidence chain and scope statement — every gateable DUT here is
   azimuthally symmetric, so this battery alone still excludes the
   transition-discontinuity class issue #489 targets; that class is covered,
   at EXPERIMENTAL, diagnostic-only status, by ``compute_coax_msl_transition``).
   Split, per this repo's physics-run discipline:

   * FAST (no FDTD): the two-drive assembly logic
     (``_assemble_coaxial_two_port_from_voltages``) is exercised with PLANTED
     analytic V(z) values built from a KNOWN asymmetric, reciprocal S-matrix,
     so the a/b sign convention derived in ``docs/design_notes/
     i489_stage2_two_port_fdtd_predeclaration.md`` is checked before any FDTD
     time is spent. Plus: layout/fit-check error paths, registration-rejection
     error paths, and a firing/non-firing control on the passivity-advisory
     routing (design-note incidental defect 3: the 1-port result bypasses that
     check; this result must not).
   * SLOW (``slow_physics``, real FDTD, its own process): the full
     through-line fixture. Gates are pinned from the MEASURED run, not
     pre-declared — see that test's docstring for the run that produced them.

2. The two-drive 2x2 solve ``solve_two_port_from_wave_amplitudes`` — pure, no
   FDTD — was ``test_coax_two_port_solve.py``. Pins the design decision as
   executable facts rather than prose:

   * the obvious rule ``S[j,i] = b_j / a_i`` has a HARD floor set by the
     terminator (measured 0.0800 with rfx's validated |Gamma_t| < 0.08
     terminator), so it cannot be made accurate by improving the
     implementation;
   * the two-drive solve removes that floor entirely, staying exact to ~1e-13
     even at |Gamma_t| = 0.5;
   * which implementation defects the local witnesses DO and DO NOT separate.
     Two of the three families are invisible to both witnesses here, and that
     is recorded as an explicit gap rather than left for someone to discover:
     a per-port unit-modulus factor (sign flip / reference-plane shift) is a
     diagonal similarity, and a systematic both-drive a/b mislabel is a
     non-diagonal one that preserves |S21| = |S12| exactly. Only PASSIVITY
     catches the latter.

   All fields are planted analytically from a signal-flow solve, so the
   "truth" there is exact and independent of any rfx extraction path.

The AD gate for this method stays in
``tests/unit/autodiff/test_coax_two_port_ad.py`` (gradient question → autodiff,
tier-4b rule); the single-plane coaxial S-matrix lane and its plumbing guards
were removed in #1212. Every assertion, tolerance, fixture value
and parametrisation of the absorbed files is kept verbatim.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rfx import Box
from rfx.api import Simulation
from rfx.api._sparams import (
    _assemble_coaxial_two_port_from_voltages,
    _finalize_sparam_result,
)
from rfx.api._spec import CoaxialTwoPortResult
from rfx.sources.coaxial_port import solve_two_port_from_wave_amplitudes
from rfx.sources.sources import GaussianPulse
from tests._wave_convention import plant_ladder_voltages_physical



BAND = np.array([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


class _NoWarningsCtx:
    """Context manager that fails the test if any warning is emitted inside it."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        return False


def _no_warnings_ctx():
    return _NoWarningsCtx()


def _sim(**kw):
    domain = kw.pop("domain", (0.008, 0.008, 0.060))
    freq_max = kw.pop("freq_max", 40.0e9)
    boundary = kw.pop("boundary", "cpml")
    sim = Simulation(domain=domain, freq_max=freq_max, boundary=boundary, **kw)
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim



def _plant_ab(s_true, gamma_t, n_f):
    """Wave amplitudes at both ports for both drives, terminator ``gamma_t``.

    Identical signal-flow construction to
    ``tests/unit/sparams/test_coax_two_port_smatrix.py::_plant`` (same convention:
    ``a[j, i]`` / ``b[j, i]`` indexed [measured_port, driven_port, freq]).
    ``gamma_t`` here plays the same physical role as each port's own feed's
    residual self-mismatch in the real fixture.
    """
    s11, s12, s21, s22 = s_true
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    b2 = s21 / (1.0 - s22 * gamma_t)
    a[0, 0], a[1, 0] = 1.0, gamma_t * b2
    b[1, 0], b[0, 0] = b2, s11 + s12 * (gamma_t * b2)
    b1 = s12 / (1.0 - s11 * gamma_t)
    a[1, 1], a[0, 1] = 1.0, gamma_t * b1
    b[0, 1], b[1, 1] = b1, s22 + s21 * (gamma_t * b1)
    return a, b


_DUT_SIGN_TOP = -1.0
_DUT_SIGN_BOT = +1.0

_Z_PLANES_TOP_M = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
_Z_PLANES_BOT_M = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
_REF_TOP_M = 0.17
_REF_BOT_M = -0.02


def _plant_two_port_voltages(a, b, *, gamma):
    """(v_top_by_drive, v_bot_by_drive) for the planted two-port fixture."""
    v_top = np.stack([
        plant_ladder_voltages_physical(
            a[0, drive], b[0, drive], gamma=gamma,
            planes_m=_Z_PLANES_TOP_M, ref_m=_REF_TOP_M, dut_sign=_DUT_SIGN_TOP,
        ) for drive in range(2)
    ], axis=0)
    v_bot = np.stack([
        plant_ladder_voltages_physical(
            a[1, drive], b[1, drive], gamma=gamma,
            planes_m=_Z_PLANES_BOT_M, ref_m=_REF_BOT_M, dut_sign=_DUT_SIGN_BOT,
        ) for drive in range(2)
    ], axis=0)
    return v_top, v_bot


def _s_mat_per_freq(s_true, n_f):
    return [
        np.array([[np.atleast_1d(s_true[0])[fi], np.atleast_1d(s_true[1])[fi]],
                  [np.atleast_1d(s_true[2])[fi], np.atleast_1d(s_true[3])[fi]]])
        for fi in range(n_f)
    ]


def test_the_planted_dut_signs_are_the_fixture_geometry():
    """``dut_sign`` is read off the fixture, not tuned.

    Both feeds sit exterior of their own probe array, so on BOTH arrays the
    reference plane is on the far side of the probes from the through line:
    the wave travelling toward it is LEAVING the DUT, which is why
    ``_assemble_coaxial_two_port_from_voltages`` ships ``a = backward_amp``
    (its Notes: the constant "is NOT a general fact of the extractor").
    This test pins the geometry that licenses it; the constant itself is
    pinned by ``test_planted_voltages_recover_known_asymmetric_s_matrix``,
    which plants from this geometry and never from the extractor's labels.
    """
    assert _REF_TOP_M > _Z_PLANES_TOP_M.max()
    assert _REF_BOT_M < _Z_PLANES_BOT_M.min()
    assert _DUT_SIGN_TOP == -np.sign(_REF_TOP_M - _Z_PLANES_TOP_M.mean())
    assert _DUT_SIGN_BOT == -np.sign(_REF_BOT_M - _Z_PLANES_BOT_M.mean())


def test_planted_fields_are_unchanged_by_the_frozen_planting_contract():
    """The frozen geometric planting reproduces the retired label-blind one.

    Guards the ONE thing that could have moved a committed
    ``compute_coaxial_two_port`` number when the #822 convention contract
    was adopted here: the planted FIELD. The retired construction is
    reproduced inline (it exists nowhere else now) and compared to the
    frozen contract on this fixture's own geometry.
    Earlier measured differences between the retired and frozen planting: 4.30e-16 / 2.78e-16 / 5.47e-16 / 7.77e-16.
    """
    def _retired_label_blind_planting(a, b, *, gamma, z_planes_m, ref_m, load_below):
        """The pre-#822 helper, verbatim in algebra: invert the extractor's
        own extrapolation so the planting agrees with whatever the assembler
        reads. Retained ONLY as the numerical witness below."""
        z_planes_m = np.asarray(z_planes_m, dtype=np.float64)
        a = np.atleast_1d(np.asarray(a, dtype=np.complex128))
        b = np.atleast_1d(np.asarray(b, dtype=np.complex128))
        gamma = np.broadcast_to(np.asarray(gamma, dtype=np.complex128), a.shape)
        z0 = float(z_planes_m.mean())
        zr = ref_m - z0
        if load_below:
            A = b * np.exp(-gamma * zr)
            B = a * np.exp(+gamma * zr)
        else:
            A = a * np.exp(-gamma * zr)
            B = b * np.exp(+gamma * zr)
        zc = z_planes_m - z0
        return (np.exp(np.multiply.outer(zc, gamma)) * A[None, :]
                + np.exp(np.multiply.outer(zc, -gamma)) * B[None, :])

    n_f = 3
    s_true = (
        np.array([0.15 + 0.05j, 0.20 - 0.03j, -0.05 + 0.10j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),
        np.array([-0.10 + 0.08j, 0.05 - 0.12j, 0.15 + 0.02j]),
    )
    a_true, b_true = _plant_ab(s_true, 0.05, n_f)
    gamma = 1j * np.array([40.0, 50.0, 65.0])
    for idx, planes, ref, load_below, dut_sign in (
        (0, _Z_PLANES_TOP_M, _REF_TOP_M, False, _DUT_SIGN_TOP),
        (1, _Z_PLANES_BOT_M, _REF_BOT_M, True, _DUT_SIGN_BOT),
    ):
        for drive in range(2):
            old = _retired_label_blind_planting(
                a_true[idx, drive], b_true[idx, drive], gamma=gamma,
                z_planes_m=planes, ref_m=ref, load_below=load_below)
            new = plant_ladder_voltages_physical(
                a_true[idx, drive], b_true[idx, drive], gamma=gamma,
                planes_m=planes, ref_m=ref, dut_sign=dut_sign)
            assert np.abs(old - new).max() <= 1e-15, np.abs(old - new).max()


def test_planted_voltages_recover_known_asymmetric_s_matrix():
    """The full post-FDTD assembly recovers a KNOWN asymmetric reciprocal S.

    Pins the a/b sign convention derived in ``docs/design_notes/
    i489_stage2_two_port_fdtd_predeclaration.md`` for
    ``compute_coaxial_two_port``'s specific dual-duty-feed geometry: for
    BOTH ports, ``a = backward_amp`` and ``b = forward_amp``. The fixture is
    deliberately ASYMMETRIC (S11 != S22) — a symmetric through-line fixture
    cannot discriminate a systematic a/b mislabel at all (see
    ``test_coax_two_port_solve.py::
    test_both_drive_swap_gap_requires_the_downstream_passivity_handle``), so
    this is the one witness capable of catching that defect family before
    any FDTD time is spent. ``s_true`` and ``gamma`` are DIFFERENT at each
    of the 3 planted frequencies (not the same value broadcast via
    ``np.full``) so a per-frequency indexing bug in the assembly loop (e.g.
    a transposed or fixed frequency index) would show up as a mismatch
    rather than passing by coincidence.
    The passivity sanity check (not required by the algebra under test) keeps the fixture physically plausible at every one of the 3 distinct frequency points.
    """
    n_f = 3
    s_true = (
        np.array([0.15 + 0.05j, 0.20 - 0.03j, -0.05 + 0.10j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),
        np.array([-0.10 + 0.08j, 0.05 - 0.12j, 0.15 + 0.02j]),
    )
    assert np.array_equal(s_true[2], s_true[1]), "Planted truth must have S21 == S12 (reciprocal)"
    s_mat_per_freq = [
        np.array([[s_true[0][fi], s_true[1][fi]], [s_true[2][fi], s_true[3][fi]]])
        for fi in range(n_f)
    ]
    for s_mat_fi in s_mat_per_freq:
        assert np.all(np.linalg.svd(s_mat_fi, compute_uv=False) <= 1.0)

    FEED_RESIDUAL_SELF_MISMATCH = 0.05
    gamma_t = FEED_RESIDUAL_SELF_MISMATCH
    a_true, b_true = _plant_ab(s_true, gamma_t, n_f)

    BETA_FIRST_RAD_PER_M = 40.0
    BETA_SECOND_RAD_PER_M = 50.0
    BETA_THIRD_RAD_PER_M = 65.0
    gamma = 1j * np.array([BETA_FIRST_RAD_PER_M, BETA_SECOND_RAD_PER_M, BETA_THIRD_RAD_PER_M])
    assert np.all(gamma.real == 0) and np.unique(gamma.imag).size == n_f, "Synthetic propagation must be lossless with a distinct beta at every frequency"
    v_top_by_drive, v_bot_by_drive = _plant_two_port_voltages(
        a_true, b_true, gamma=gamma)

    s_params, cond_a, rec_resid, fit_resid, gamma_fit = _assemble_coaxial_two_port_from_voltages(
        z_planes_bot_m=_Z_PLANES_BOT_M, z_planes_top_m=_Z_PLANES_TOP_M,
        ref_bot_m=_REF_BOT_M, ref_top_m=_REF_TOP_M,
        v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
    )

    for fi in range(n_f):
        np.testing.assert_allclose(s_params[:, :, fi], s_mat_per_freq[fi], atol=1e-9)
    assert np.all(np.isfinite(rec_resid)) and np.all(rec_resid < 1e-9)
    assert np.all(np.isfinite(fit_resid)) and np.all(fit_resid < 1e-9)
    assert np.all(cond_a < 3.0)
    np.testing.assert_allclose(gamma_fit, np.broadcast_to(gamma, gamma_fit.shape), atol=1e-6)


def test_swapped_ab_convention_fails_the_same_asymmetric_fixture():
    """Non-firing control's mirror: the WRONG convention must NOT also pass,
    and it must fail in the ONE way the algebra predicts.

    Plant both ladders on the wrong side of their reference planes
    (``dut_sign`` flipped at both ports) and the assembler's incident and
    outgoing roles are exchanged at both, so ``S = B inv(A)`` with ``A`` and
    ``B`` swapped -- exactly ``inv(S_true)``.

    Was ``assert not np.allclose(...)`` on a single frequency, which fires
    for ANY discrepancy including a fit that merely fell apart. Strengthened
    to the exact inverse over all three bins as part of the #822 convention
    contract: measured max|S - inv(S_true)| = 3.25e-15 / 2.32e-15 / 3.29e-15,
    against max|S - S_true| = 0.688 / 1.19 / 1.37 (so the control genuinely
    fires -- ``inv(S_true) != S_true`` by O(1) at every bin).
    """
    n_f = 3
    s_true = (
        np.array([0.15 + 0.05j, 0.20 - 0.03j, -0.05 + 0.10j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),
        np.array([-0.10 + 0.08j, 0.05 - 0.12j, 0.15 + 0.02j]),
    )
    a_true, b_true = _plant_ab(s_true, 0.05, n_f)
    gamma = 1j * np.array([40.0, 50.0, 65.0])

    v_top_by_drive = np.stack([
        plant_ladder_voltages_physical(
            a_true[0, drive], b_true[0, drive], gamma=gamma,
            planes_m=_Z_PLANES_TOP_M, ref_m=_REF_TOP_M, dut_sign=-_DUT_SIGN_TOP,
        ) for drive in range(2)
    ], axis=0)
    v_bot_by_drive = np.stack([
        plant_ladder_voltages_physical(
            a_true[1, drive], b_true[1, drive], gamma=gamma,
            planes_m=_Z_PLANES_BOT_M, ref_m=_REF_BOT_M, dut_sign=-_DUT_SIGN_BOT,
        ) for drive in range(2)
    ], axis=0)

    s_params, *_ = _assemble_coaxial_two_port_from_voltages(
        z_planes_bot_m=_Z_PLANES_BOT_M, z_planes_top_m=_Z_PLANES_TOP_M,
        ref_bot_m=_REF_BOT_M, ref_top_m=_REF_TOP_M,
        v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
    )
    for fi, s_mat_true in enumerate(_s_mat_per_freq(s_true, n_f)):
        s_inv = np.linalg.inv(s_mat_true)
        np.testing.assert_allclose(s_params[:, :, fi], s_inv, atol=1e-9)
        assert np.abs(s_inv - s_mat_true).max() > 0.5



def test_compute_coaxial_two_port_drive_index_matches_physical_port(monkeypatch):
    """Pins compute_coaxial_two_port's own call-site wiring: drive_idx=0 in
    ``v_bot_by_drive``/``v_top_by_drive`` must hold data from PHYSICALLY
    driving port 1 (the top, +z end); drive_idx=1 must hold port 2 (bottom,
    -z end) -- the contract ``_assemble_coaxial_two_port_from_voltages``'s
    own docstring states ("index 0 = port 1 driven, index 1 = port 2
    driven") but that the call site itself had never been checked against.

    METHOD: stub out the expensive/irrelevant physics -- the FDTD run
    (``rfx.simulation.run``) and the field -> voltage extraction
    (``rfx.sources.coaxial_port.coaxial_line_plane_voltage``) -- and
    intercept ``_assemble_coaxial_two_port_from_voltages`` to capture
    exactly what the real call site (unmodified) passed as
    ``v_bot_by_drive``/``v_top_by_drive``. The stubbed FDTD run identifies
    which PHYSICAL port it was asked to excite purely from the z-index of
    the source cells it received (top half of the domain = physically the
    port-1/+z end, bottom half = physically the port-2/-z end) -- a
    criterion entirely INDEPENDENT of the call site's own ``drive_idx``
    numbering -- and returns a sentinel keyed by that physical identity.
    The real ``for drive_idx, spec in enumerate((spec_top, spec_bot))``
    loop, the real per-drive ``_run(...)`` call, and the real per-plane
    ``coaxial_line_plane_voltage(...)`` call sites all execute unmodified;
    only the physics underneath each is replaced.

    TWO-AXIS SENTINEL (N6, review fix): the sentinel encodes BOTH which
    physical drive excited it (real part) AND which probe ARRAY position
    in ``result.dft_planes`` it was read from -- bot-block (first half of
    the list) vs top-block (second half) -- via the imaginary part. A
    drive-only sentinel (the first version of this test) would NOT catch
    a separate defect class: a ``top_off`` indexing bug in the call site
    (``rfx/api/_sparams.py``, the ``top_off = n_bot * 2`` slice offset)
    that reads the WRONG half of ``result.dft_planes`` for the top array
    would still receive the SAME (drive-keyed only) sentinel value the
    bot array got, and this test would pass vacuously. With a two-axis
    sentinel, that class of bug flips ``v_bot_by_drive``'s array-axis
    component to the "top" value (or vice versa), which the assertions
    below catch independently of the drive-axis check. VERIFIED BY HAND
    (second swap experiment, N6): temporarily setting ``top_off = 0``
    instead of ``n_bot * 2`` turned this test RED (``v_top[0]`` read
    ``1+10j``, the bot-block sentinel, instead of the expected ``1+20j``)
    -- confirmed, then reverted.

    SWAP EXPERIMENT (performed by hand for this change, per the R3
    discipline -- not merely asserted): with ``rfx/api/_sparams.py``
    unmodified this test is GREEN. Temporarily editing the call site's
    ``enumerate((spec_top, spec_bot))`` to ``enumerate((spec_bot, spec_top))``
    (physically swapping which drive runs first) and rerunning this test
    turns it RED (drive_idx=0 then captures the port-2 sentinel instead of
    port-1's) -- confirmed, then reverted back to the original order before
    this change was committed.

    NOTE ON WHAT THIS DOES AND DOES NOT PROTECT: the recovered S-parameter
    MATRIX itself is provably invariant to this specific swap when the
    measured-port row assignment stays fixed (which it does here --
    ``v_top_by_drive``/``v_bot_by_drive`` are always filled from the
    top/bottom probe ARRAYS respectively, regardless of drive_idx or which
    spec is active): swapping the drive_idx column labeling permutes both
    ``a_inc`` and ``b_out``'s columns by the SAME permutation matrix P, and
    ``S = B @ inv(A) = (B@P) @ inv(A@P) = (B@P) @ (inv(P)@inv(A)) =
    B@(P@inv(P))@inv(A) = B@inv(A)`` -- unchanged, because P is its own
    inverse. So ``res.s_params`` from ``test_matched_through_line_
    transmits_reciprocally`` (the slow_physics fixture) CANNOT see this
    defect class even in principle, and neither can ``cond_a`` (the
    condition number of a 2x2 matrix is invariant to column permutation)
    or the R5 mechanism gate's ``gamma_bar`` (a mean over the drive axis,
    also permutation-invariant). Only the PER-DRIVE ``gamma``/
    ``recurrence_residual``/``fit_residual`` diagnostics, and the
    ``port_names``/physical-location contract, would silently scramble --
    which is exactly why this test inspects the intermediate arrays
    directly instead of trusting the final S matrix.
    Patching the ``rfx.api._sparams`` re-export would leave ``captured`` empty.
    """
    sim = _sim()
    grid = sim._build_grid()
    nz_mid = grid.shape[2] / 2.0

    captured: dict = {}

    def _fake_assemble(**kwargs):
        captured["v_bot_by_drive"] = np.array(kwargs["v_bot_by_drive"])
        captured["v_top_by_drive"] = np.array(kwargs["v_top_by_drive"])
        n_f = captured["v_bot_by_drive"].shape[-1]
        s = np.full((2, 2, n_f), 0.1 + 0.0j, dtype=np.complex128)
        return (
            s,
            np.ones(n_f),
            np.zeros((2, 2, n_f)),
            np.zeros((2, 2, n_f)),
            np.zeros((2, 2, n_f), dtype=complex),
        )

    class _FakeAcc:
        def __init__(self, v):
            self.accumulator = v

    class _FakeResult:
        def __init__(self, dft_planes, time_series, dt):
            self.dft_planes = dft_planes
            self.time_series = time_series
            self.dt = dt

    DRIVE_PORT1, DRIVE_PORT2 = 1.0, 2.0
    ARRAY_BOT, ARRAY_TOP = 10.0, 20.0

    def _fake_run(grid_, materials, n_steps, *, boundary, cpml_axes, sources,
                 mag_sources, probes, dft_planes, return_state,
                 pec_edge_masks=None, record_dft=False, measurement=None):
        """The explicit runner signature makes a new argument at the real call site raise TypeError instead of being silently absorbed."""
        assert record_dft is False and measurement is not None
        k0 = float(sources[0].k)
        drive_val = DRIVE_PORT1 if k0 > nz_mid else DRIVE_PORT2
        n_total = len(dft_planes)
        n_bot_block = n_total // 2
        fake_planes = []
        for i in range(n_total):
            if i % 2 == 0:
                array_val = ARRAY_BOT if i < n_bot_block else ARRAY_TOP
                sentinel = drive_val + array_val * 1j
                fake_planes.append(_FakeAcc(np.array([sentinel])))
            else:
                fake_planes.append(_FakeAcc(np.array([0.0j])))
        ts = np.full((10, len(probes)), 1e-6)
        return _FakeResult(fake_planes, ts, grid_.dt)

    def _fake_voltage(grid_, ex_dft, ey_dft, *, center_xy, pin_radius, outer_radius, **kwargs):
        return np.asarray(ex_dft, dtype=np.complex128)

    import rfx.simulation as _simulation_mod
    import rfx.sources.coaxial_port as _coax_mod
    import rfx.sparams.coax as _coax_lane_mod

    monkeypatch.setattr(_simulation_mod, "run", _fake_run)
    monkeypatch.setattr(_coax_mod, "coaxial_line_plane_voltage", _fake_voltage)
    monkeypatch.setattr(
        _coax_lane_mod, "_assemble_coaxial_two_port_from_voltages", _fake_assemble
    )

    sim.compute_coaxial_two_port(n_steps=10, freqs=np.array([8.0e9]), probe_count=12)

    v_bot = captured["v_bot_by_drive"]
    v_top = captured["v_top_by_drive"]
    assert v_bot.shape[0] == 2 and v_top.shape[0] == 2
    np.testing.assert_allclose(v_bot[0], DRIVE_PORT1 + ARRAY_BOT * 1j)
    np.testing.assert_allclose(v_top[0], DRIVE_PORT1 + ARRAY_TOP * 1j)
    np.testing.assert_allclose(v_bot[1], DRIVE_PORT2 + ARRAY_BOT * 1j)
    np.testing.assert_allclose(v_top[1], DRIVE_PORT2 + ARRAY_TOP * 1j)



def test_boundary_must_be_cpml():
    sim = _sim(boundary="pec")
    with pytest.raises(ValueError, match="requires boundary='cpml'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_absorbing_on_z_only_is_refused():
    """The line's two ends are unshielded; a z-only absorber left them in a
    closed PEC can (issue 1218). The full contract, with a runner spy, is
    ``tests/unit/sparams/test_coax_lanes_absorb_on_every_axis.py``."""
    sim = _sim()
    with pytest.raises(ValueError, match="accepts only cpml_axes='xyz'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1, cpml_axes="z")


def test_periodic_axes_rejected():
    """A periodic lateral axis has no absorber, so the all-six-faces absorber requirement is the refusal that fires first."""
    from rfx.boundaries.spec import BoundarySpec
    sim = _sim(boundary=BoundarySpec(x="periodic", y="cpml", z="cpml"))
    with pytest.raises(ValueError, match="positive CPML thickness on all six faces"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


@pytest.mark.parametrize(
    "profile_kw",
    [{"dx_profile": np.full(8, 1.0e-3)}, {"dz_profile": np.full(60, 1.0e-3)}],
    ids=("dx_profile", "dz_profile"),
)
def test_nonuniform_profiles_rejected(profile_kw):
    sim = _sim(dx=1.0e-3, **profile_kw)
    with pytest.raises(ValueError, match="only a uniform Yee grid"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_existing_tfsf_rejected():
    sim = _sim()
    sim.add_tfsf_source(f0=8.0e9)
    with pytest.raises(ValueError, match="existing TFSF source"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_refinement_rejected():
    sim = _sim()
    sim.add_refinement((0.018, 0.022), ratio=2, validation="research")
    with pytest.raises(ValueError, match="does not support SBP-SAT refinement"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)




def test_mixed_precision_rejected():
    sim = _sim(precision="mixed")
    with pytest.raises(ValueError, match="requires precision='float32'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_fourth_order_stencil_rejected():
    sim = _sim(stencil_order=4)
    with pytest.raises(ValueError, match="requires stencil_order=2"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_two_dimensional_mode_rejected():
    sim = _sim(mode="2d_tmz")
    with pytest.raises(ValueError, match="requires mode='3d'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("feature", ("geometry", "thin_conductor"))
@pytest.mark.parametrize("method", ("compute_coaxial_two_port", "compute_coaxial_line_reflection"))
def test_registered_geometry_rejected(feature, method, monkeypatch):
    sim = _sim()
    shape = Box((0.001, 0.001, 0.010), (0.002, 0.002, 0.011))
    if feature == "geometry":
        sim.add_material("test_dielectric", eps_r=2.0)
        sim.add(shape, material="test_dielectric")
    else:
        sim.add_thin_conductor(shape, sigma_bulk=1.0e4, thickness=35.0e-6)

    def unexpected_resolution():
        pytest.fail("forbidden declarations must be rejected before mesh planning")

    monkeypatch.setattr(sim, "_auto_configure_mesh", unexpected_resolution)
    with pytest.raises(ValueError, match="constructs the complete line geometry"):
        getattr(sim, method)(n_steps=1, n_freqs=1)


def test_lumped_rlc_rejected():
    sim = _sim()
    sim.add_lumped_rlc((0.004, 0.004, 0.010), R=50.0, topology="parallel")
    with pytest.raises(ValueError, match="registered lumped RLC elements"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("monitor", ("probe", "dft", "flux", "ntff"))
def test_registered_monitor_rejected(monitor):
    sim = _sim()
    if monitor == "probe":
        sim.add_probe((0.004, 0.004, 0.020))
    elif monitor == "dft":
        sim.add_dft_plane_probe(axis="z", coordinate=0.020, n_freqs=1)
    elif monitor == "flux":
        sim.add_flux_monitor(axis="z", coordinate=0.020, n_freqs=1)
    else:
        sim.add_ntff_box((0.001, 0.001, 0.010), (0.007, 0.007, 0.030), n_freqs=1)
    with pytest.raises(ValueError, match="does not consume registered"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("probe_count", (0, 1, 2))
def test_at_least_three_probe_planes_required(probe_count):
    sim = _sim()
    with pytest.raises(ValueError, match="probe_count must be at least 3"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1, probe_count=probe_count)


@pytest.mark.parametrize("probe_count", (True, 3.5))
def test_probe_count_must_be_an_integer(probe_count):
    sim = _sim()
    with pytest.raises(ValueError, match="probe_count must be an integer"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1, probe_count=probe_count)


def test_no_coaxial_port_rejected():
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary="cpml")
    with pytest.raises(ValueError, match="exactly one add_coaxial_port"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_two_coaxial_ports_rejected():
    sim = _sim()
    sim.add_coaxial_port((0.004, 0.004, 0.040), face="top")
    with pytest.raises(ValueError, match="exactly one add_coaxial_port"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_non_top_face_rejected():
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="bottom",
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    with pytest.raises(NotImplementedError, match="requires the registered port's face='top'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)



def test_domain_too_short_for_two_feed_layout_is_rejected():
    """A tiny z domain cannot fit both ends' fixed (2/1/3-cell) offsets plus their probe arrays."""
    sim = _sim(domain=(0.008, 0.008, 0.006))
    with pytest.raises(ValueError, match="domain too short|probe arrays overlap"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_overlapping_probe_arrays_are_rejected():
    """The domain has enough room for the fixed offsets but not for two 12-plane, 4-cell spaced arrays without touching."""
    sim = _sim(domain=(0.008, 0.008, 0.020))
    with pytest.raises(ValueError, match="probe arrays overlap|domain too short"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_default_domain_fits_the_default_layout():
    """Non-firing control: the domain this suite's slow test uses must NOT
    trip either fit-check — i.e. the two rejection tests above are testing
    a real constraint, not a constraint that always fires."""
    sim = _sim()
    grid = sim._build_grid()
    nz = grid.shape[2]
    z_hi_coax_top = nz - int(grid.pad_z_hi) - 2
    z_src_top = z_hi_coax_top - 3
    z_lo_coax_bot = int(grid.pad_z_lo) + 2
    z_src_bot = z_lo_coax_bot + 3
    probes_top = sorted(z_src_top - 8 - 4 * k for k in range(12))
    probes_bot = sorted(z_src_bot + 8 + 4 * k for k in range(12))
    assert probes_bot[-1] < probes_top[0]
    assert z_lo_coax_bot < z_hi_coax_top


def test_default_coax_fixture_realizes_the_declared_mesh_and_absorber_depth():
    """What lattice this coax fixture actually gets, in absolute numbers.

    The declaration is a 8 x 8 x 60 mm domain driven to 40 GHz with an
    absorbing boundary, and it names neither a cell size nor an absorber
    depth; rfx chooses both. The cell comes out a twentieth of the shortest
    free-space wavelength, 0.37474057 mm, and every one of the six faces
    gets 16 CPML cells. That leaves 22 x 22 x 161 cells of clear interior
    inside a grid of 55 x 55 x 194 nodes.

    Pinned against literals rather than against the expressions that produce
    them, so a change in the auto-mesh rule, in the default absorber depth
    or in the node/cell fencepost moves this test. It came here from the
    coax thru-line case's header suite when that case was removed
    (2026-09-22) and is the only place these three defaults are asserted
    together for a coaxial-port simulation; the values are unchanged.
    """
    grid = _sim()._build_grid()

    assert grid.shape == (55, 55, 194)
    assert grid.dx == pytest.approx(3.7474057249999997e-4, rel=1e-12)
    assert grid.dx == pytest.approx(299792458.0 / 40.0e9 / 20.0, rel=1e-12)

    assert (grid.pad_x_lo, grid.pad_x_hi) == (16, 16)
    assert (grid.pad_y_lo, grid.pad_y_hi) == (16, 16)
    assert (grid.pad_z_lo, grid.pad_z_hi) == (16, 16)

    interior = tuple(grid.shape[a] - 1 - (16 + 16) for a in range(3))
    assert interior == (22, 22, 161)



def _dummy_result(s_params):
    n_f = s_params.shape[-1]
    return CoaxialTwoPortResult(
        s_params=s_params,
        freqs=np.linspace(4e9, 12e9, n_f),
        port_names=("port1", "port2"),
        reference_planes=np.array([0.05, 0.001]),
        cond_a=np.ones(n_f),
        recurrence_residual=np.zeros((2, 2, n_f)),
        fit_residual=np.zeros((2, 2, n_f)),
        gamma=np.zeros((2, 2, n_f), dtype=complex),
        annulus_cells=4.0,
        settling_db=np.array([-45.0, -45.0]),
        status="passed",
    )


def test_passive_result_does_not_warn_non_firing_control():
    s_true = np.array(
        [[0.05 + 0.0j, 0.95 + 0.0j], [0.95 + 0.0j, 0.05 + 0.0j]]
    )[:, :, None]
    result = _dummy_result(s_true)
    with _no_warnings_ctx():
        out = _finalize_sparam_result(result, extractor="compute_coaxial_two_port", strict=False)
    assert out is result


def test_nonpassive_result_warns_and_is_not_silently_dropped():
    s_bad = np.array(
        [[0.05 + 0.0j, 5.0 + 0.0j], [5.0 + 0.0j, 0.05 + 0.0j]]
    )[:, :, None]
    assert np.all(np.sum(np.abs(s_bad) ** 2, axis=0) > 1.0), "Column power must exceed one for the non-passive fixture"
    result = _dummy_result(s_bad)
    with pytest.warns(UserWarning, match="passivity"):
        out = _finalize_sparam_result(result, extractor="compute_coaxial_two_port", strict=False)
    assert out is result


@pytest.mark.source_scan
def test_compute_coaxial_two_port_routes_through_finalize_sparam_result():
    """Regression guard for design-note incidental defect 3: greppable proof
    the new method's own source calls the shared passivity-advisory epilogue
    (unlike compute_coaxial_line_reflection, which returns
    CoaxialLineReflectionResult directly)."""
    src = inspect.getsource(Simulation.compute_coaxial_two_port)
    assert "_finalize_sparam_result(" in src


@pytest.mark.slow_physics
def test_matched_through_line_transmits_reciprocally(monkeypatch):
    """The fit_residual < 0.02 number is borrowed from the recurrence gate for convenience (same repo, same 2-decimal round number, ~1.6x margin over the earlier measured max 0.0127)."""
    sim = _sim()
    import rfx.sparams.coax as coax

    captures = []
    original = coax.renormalize_power_waves

    def record(s_params, line_impedances, reference_impedances):
        captures.append(tuple(np.array(x, copy=True) for x in (
            s_params, line_impedances, reference_impedances,
        )))
        return original(s_params, line_impedances, reference_impedances)

    monkeypatch.setattr(coax, "renormalize_power_waves", record)
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_coaxial_two_port(n_steps=6000, freqs=BAND)
    assert [str(w.message) for w in caught] == []

    assert res.status == "passed"
    assert res.annulus_cells >= 3.5
    assert np.all(np.isfinite(res.s_params))
    THROUGH_LINE_CONDITION_LIMIT = 3.0
    assert np.all(res.cond_a < THROUGH_LINE_CONDITION_LIMIT)
    RECURRENCE_RESIDUAL_LIMIT = 0.02
    assert np.all(res.recurrence_residual < RECURRENCE_RESIDUAL_LIMIT)
    FIT_RESIDUAL_LIMIT = 0.02
    assert np.all(res.fit_residual < FIT_RESIDUAL_LIMIT)
    SETTLING_LIMIT_DB = -40.0
    assert np.all(res.settling_db < SETTLING_LIMIT_DB)

    S = res.s_params
    s11, s21, s12, s22 = S[0, 0], S[1, 0], S[0, 1], S[1, 1]
    TRANSMISSION_MAGNITUDE_FLOOR = 0.70
    assert np.all(np.abs(s21) >= TRANSMISSION_MAGNITUDE_FLOOR), np.abs(s21)
    assert np.all(np.abs(s12) >= TRANSMISSION_MAGNITUDE_FLOOR), np.abs(s12)
    assert len(captures) == 1
    native, line_impedances, reference_impedances = captures[0]
    NATIVE_REFLECTION_MAGNITUDE_CEILING = 0.08
    assert np.all(np.abs(native[0, 0]) <= NATIVE_REFLECTION_MAGNITUDE_CEILING), np.abs(native[0, 0])
    assert np.all(np.abs(native[1, 1]) <= NATIVE_REFLECTION_MAGNITUDE_CEILING), np.abs(native[1, 1])

    np.testing.assert_array_equal(line_impedances, res.line_impedances)
    np.testing.assert_array_equal(reference_impedances, [50.0, 50.0])
    port = sim._coaxial_ports[0]
    mu0, eps0, eps_r = 1.25663706212e-6, 8.8541878128e-12, 2.1
    drawn_impedance = np.sqrt(mu0 / (eps0 * eps_r)) * np.log(
        float(port.outer_radius) / float(port.pin_radius)
    ) / (2.0 * np.pi)
    assert np.all(np.abs(line_impedances - drawn_impedance) > 1.0), (
        line_impedances, drawn_impedance,
    )

    rho_b = (50.0 - line_impedances) / (50.0 + line_impedances)
    R = np.diag(rho_b)
    A = np.diag(np.sqrt(1.0 - rho_b ** 2))
    model = np.stack([
        np.linalg.inv(A) @ (native[:, :, fi] - R)
        @ np.linalg.inv(np.eye(2) - R @ native[:, :, fi]) @ A
        for fi in range(len(BAND))
    ], axis=-1)
    np.testing.assert_allclose(S, model, rtol=0.0, atol=1e-4)

    RECIPROCITY_MAGNITUDE_TOLERANCE = 0.01
    assert np.all(np.abs(np.abs(s21) / np.abs(s12) - 1.0) < RECIPROCITY_MAGNITUDE_TOLERANCE)
    phase_diff_deg = np.degrees(np.angle(s21) - np.angle(s12))
    RECIPROCITY_PHASE_TOLERANCE_DEG = 1.0
    assert np.all(np.abs(phase_diff_deg) < RECIPROCITY_PHASE_TOLERANCE_DEG), phase_diff_deg

    L12 = float(res.reference_planes[0] - res.reference_planes[1])
    gamma_bar = np.mean(res.gamma.real, axis=(0, 1))
    compensated_s21 = np.abs(s21) * np.exp(gamma_bar * L12)
    assert np.all((compensated_s21 >= 0.95) & (compensated_s21 <= 1.05)), (
        compensated_s21, gamma_bar, L12,
    )

    col1 = np.abs(s11) ** 2 + np.abs(s21) ** 2
    col2 = np.abs(s22) ** 2 + np.abs(s12) ** 2
    assert np.all(col1 < 1.0) and np.all(col2 < 1.0)



_FREQS = np.array([4e9, 6e9, 8e9, 10e9, 12e9])


def _plant(s_true, gamma_t, n_f):
    """Wave amplitudes at both ports for both drives, with terminator Gamma_t.

    Signal flow for a 2-port S with a terminator of reflection ``gamma_t`` at
    the NON-driven port and a unit incident wave at the driven port::

        drive 1:  a1 = 1,  a2 = gamma_t * b2
                  b2 = S21 / (1 - S22*gamma_t),  b1 = S11 + S12*b2*gamma_t

    and the mirror for drive 2. Returns ``(a, b)`` shaped
    ``(2, 2, n_f)`` indexed ``[measured_port, driven_port, freq]``.
    """
    s11, s12, s21, s22 = s_true
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    b2 = s21 / (1.0 - s22 * gamma_t)
    a[0, 0], a[1, 0] = 1.0, gamma_t * b2
    b[1, 0], b[0, 0] = b2, s11 + s12 * (gamma_t * b2)
    b1 = s12 / (1.0 - s11 * gamma_t)
    a[1, 1], a[0, 1] = 1.0, gamma_t * b1
    b[0, 1], b[1, 1] = b1, s22 + s21 * (gamma_t * b1)
    return a, b


def _through(n_f):
    """Ideal lossless through: S11=S22=0, S21=S12=1 (phase absorbed)."""
    one = np.ones(n_f, dtype=np.complex128)
    return (0.0 * one, one, one, 0.0 * one)


def _shunt_resistor(r_ohm, z0, n_f):
    """Exact S of a shunt resistor R across a line of impedance Z0."""
    one = np.ones(n_f, dtype=np.complex128)
    y = z0 / (2.0 * r_ohm)
    s11 = -y / (1.0 + y) * one
    s21 = 1.0 / (1.0 + y) * one
    return (s11, s21, s21, s11)


def _asymmetric_reciprocal(z0, n_f):
    """A reciprocal but ASYMMETRIC 2-port: series Z1, shunt Y, series Z2.

    Needed because the through line (S22 = 0) and the shunt resistor
    (S11 = S22) are both degenerate for the defect family in
    ``test_both_drive_swap_gap_requires_the_downstream_passivity_handle`` — with those
    fixtures the defect happens to collapse to a singular matrix and gets
    caught by accident rather than by design.
    """
    one = np.ones(n_f, dtype=np.complex128)
    z1, z2, y = 30.0 + 0j, 12.0 + 0j, 1.0 / (80.0 + 0j)
    a = 1.0 + z1 * y
    b = z1 + z2 + z1 * y * z2
    c = y
    d = 1.0 + y * z2
    den = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / den
    s22 = (-a + b / z0 - c * z0 + d) / den
    s21 = 2.0 / den
    s12 = 2.0 * (a * d - b * c) / den
    return (s11 * one, s12 * one, s21 * one, s22 * one)



def test_asymmetric_fixture_is_what_it_claims():
    """Verify the asymmetric DUT independently, by node analysis.

    The review that produced this fixture found the previous battery passing
    for the wrong reason: both older fixtures were DEGENERATE for the defect
    they were supposed to discriminate, and nothing said so. A fixture whose
    properties are assumed rather than checked is how that happens, so this
    one is derived a second way — plain node analysis of the series-shunt-
    series network — and compared against the ABCD formula it ships with.
    The EMF behind series Z0 is unit.
    """
    z0 = 48.5914
    z1, z2, y = 30.0, 12.0, 1.0 / 80.0
    s11, s12, s21, s22 = (v[0] for v in _asymmetric_reciprocal(z0, 1))

    zp = 1.0 / (y + 1.0 / (z2 + z0))
    zin = z1 + zp
    v1 = zin / (z0 + zin)
    v2 = (v1 * zp / (z1 + zp)) * z0 / (z2 + z0)
    zin2 = z2 + 1.0 / (y + 1.0 / (z1 + z0))
    np.testing.assert_allclose(s11.real, (zin - z0) / (zin + z0), atol=1e-12)
    np.testing.assert_allclose(s21.real, 2.0 * v2, atol=1e-12)
    np.testing.assert_allclose(s22.real, (zin2 - z0) / (zin2 + z0), atol=1e-12)

    assert np.isclose(s12, s21, atol=1e-14), "fixture must be reciprocal"
    assert not np.isclose(abs(s11), abs(s22)), "fixture must be ASYMMETRIC"
    assert abs(s11) ** 2 + abs(s21) ** 2 < 1.0, "fixture must be passive"



@pytest.mark.parametrize("gamma_t", [0.02, 0.05, 0.08])
def test_single_ratio_rule_has_a_terminator_floor(gamma_t):
    """`S[j,i] = b_j/a_i` reads |1 + S11 - S21| = |Gamma_t| EXACTLY.

    This is the measurement that killed the round-1 design: on a through line
    the whole terminator reflection lands in S11, so the through-line identity
    residual equals the terminator reflection no matter how good the rest of
    the implementation is. rfx's validated matched terminator only reaches
    |Gamma_t| < 0.08, so a 0.05-level identity gate was unreachable BY
    CONSTRUCTION rather than by any defect.
    """
    n_f = len(_FREQS)
    a, b = _plant(_through(n_f), gamma_t, n_f)
    s11_ratio = b[0, 0] / a[0, 0]
    s21_ratio = b[1, 0] / a[0, 0]
    resid = np.abs(1.0 + s11_ratio - s21_ratio)
    np.testing.assert_allclose(resid, gamma_t, rtol=1e-12)


def test_two_drive_solve_removes_the_terminator_floor():
    """The same planted fields, solved properly, are exact at every Gamma_t."""
    n_f = len(_FREQS)
    for gamma_t in (0.0, 0.02, 0.08, 0.2, 0.5):
        a, b = _plant(_through(n_f), gamma_t, n_f)
        out = solve_two_port_from_wave_amplitudes(a, b)
        np.testing.assert_allclose(np.abs(out.s_params[1, 0]), 1.0, atol=1e-13)
        np.testing.assert_allclose(np.abs(out.s_params[0, 0]), 0.0, atol=1e-13)
        assert np.all(out.cond_a < 3.0), (gamma_t, out.cond_a)


def test_conditioning_grows_slowly_with_terminator_reflection():
    """Pins the measured cond(A) trend so a regression is visible.
    The closed form is asserted because |Gt|=0.5 gives exactly 3.0 despite this platform's SVD rounding to 2.999999999999999, which a LAPACK/numpy change could flip.
    """
    n_f = 1
    conds = {}
    for gamma_t in (0.0, 0.08, 0.5):
        a, b = _plant(_through(n_f), gamma_t, n_f)
        conds[gamma_t] = float(solve_two_port_from_wave_amplitudes(a, b).cond_a[0])
    for gt, c in conds.items():
        assert c == pytest.approx((1.0 + gt) / (1.0 - gt), rel=1e-9, abs=1e-9)



def test_recovers_shunt_resistor_exactly_despite_terminator():
    """A known-mismatch DUT is recovered exactly, terminator and all."""
    n_f = len(_FREQS)
    z0, r = 48.5914, 25.0
    s_true = _shunt_resistor(r, z0, n_f)
    a, b = _plant(s_true, 0.08, n_f)
    out = solve_two_port_from_wave_amplitudes(a, b)
    np.testing.assert_allclose(out.s_params[0, 0], s_true[0], atol=1e-12)
    np.testing.assert_allclose(out.s_params[1, 0], s_true[2], atol=1e-12)
    np.testing.assert_allclose(
        out.s_params[1, 0], out.s_params[0, 1], atol=1e-12
    )



def test_swapped_receive_amplitude_one_drive_is_caught():
    """Consuming the wrong amplitude at the receive array, ONE drive only.

    The receive array holds a large transmitted wave and a small one returning
    off its terminator; taking the small one is a likely wiring mistake. This
    one-sided variant collapses |S21| and is easy to catch.

    NOTE: the realistic SYSTEMATIC version of this mistake — the same mislabel
    applied to BOTH drives — behaves completely differently and is NOT caught
    here. See the next test; do not read this one as covering that.
    """
    n_f = len(_FREQS)
    a, b = _plant(_through(n_f), 0.08, n_f)
    good = solve_two_port_from_wave_amplitudes(a, b)
    a_bad, b_bad = a.copy(), b.copy()
    b_bad[1, 0], a_bad[1, 0] = a[1, 0], b[1, 0]
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)
    assert np.all(np.abs(good.s_params[1, 0]) > 0.99)
    assert np.all(np.abs(bad.s_params[1, 0]) < 0.2), np.abs(bad.s_params[1, 0])


def test_both_drive_swap_gap_requires_the_downstream_passivity_handle():
    """DOCUMENTED GAP: the systematic a/b mislabel evades both local witnesses.

    A real implementation bug mislabels incident vs outgoing at one port
    CONSISTENTLY, i.e. for both drives. That is the non-diagonal transform
    ``S' = N M^-1`` with ``M = [[1,0],[S21,S22]]``, ``N = [[S11,S12],[0,1]]``,
    giving ``S'21 = -S21/S22`` and ``S'12 = S12/S22``. Because the two share a
    magnitude, **magnitude reciprocity is exactly blind to it**, and the
    matrix stays well conditioned, so the ill-conditioning guard is silent too.

    The through line (S22=0) and the shunt resistor (S11=S22) are DEGENERATE
    for this defect — it collapses to a singular matrix there and gets caught
    by accident. That accident is why the original battery looked complete.
    An asymmetric DUT exposes the truth.

    The independent handle is PASSIVITY, not anything in this module:
    ``S'22 = 1/S22`` explodes for any well-matched passive DUT. Whatever ships
    on top of this solve must route through the repo's passivity self-check.
    The column-power floor is well below the earlier measured column power of 317 but far above 1, so it stays meaningful if the fixture is retuned.
    """
    n_f = len(_FREQS)
    s_true = _asymmetric_reciprocal(48.5914, n_f)
    assert not np.isclose(abs(s_true[0][0]), abs(s_true[3][0]))
    a, b = _plant(s_true, 0.05, n_f)
    a_bad, b_bad = a.copy(), b.copy()
    a_bad[1, :], b_bad[1, :] = b[1, :].copy(), a[1, :].copy()
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)

    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 0]), np.abs(bad.s_params[0, 1]), atol=1e-9
    )
    ILL_CONDITIONING_SILENCE_CEILING = 1.0e2
    assert np.all(bad.cond_a < ILL_CONDITIONING_SILENCE_CEILING), bad.cond_a
    assert np.all(np.abs(bad.s_params[1, 0]) > 2.0)
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 1]), 1.0 / np.abs(s_true[3]), rtol=1e-9
    )
    col_power = np.abs(bad.s_params[0, 0]) ** 2 + np.abs(bad.s_params[1, 0]) ** 2
    assert np.all(col_power > 100.0), col_power


@pytest.mark.parametrize("scale", [1.05, 1.5, 0.8])
def test_reciprocity_catches_a_per_port_amplitude_error(scale):
    """A per-port CALIBRATION error is visible to the reciprocity witness.

    Scaling both amplitudes at one port by ``c`` is the similarity
    ``S -> D S D^-1`` with ``D = diag(1, c)``, which sends
    ``S21 -> c*S21`` and ``S12 -> S12/c``. The product is invariant but the
    RATIO moves by ``c^2``, so reciprocity sees it — including the 5%
    amplitude-calibration defect the design review flagged as untracked.
    """
    n_f = len(_FREQS)
    s_true = _shunt_resistor(25.0, 48.5914, n_f)
    a, b = _plant(s_true, 0.05, n_f)
    a_bad, b_bad = a.copy(), b.copy()
    a_bad[1, :] *= scale
    b_bad[1, :] *= scale
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)
    ratio = np.abs(bad.s_params[1, 0]) / np.abs(bad.s_params[0, 1])
    np.testing.assert_allclose(ratio, scale ** 2, rtol=1e-9)
    dev = np.abs(np.abs(bad.s_params[1, 0]) - np.abs(bad.s_params[0, 1]))
    assert np.all(dev > 0.04), dev


@pytest.mark.parametrize(
    "diag_factor", [-1.0, np.exp(-1j * 0.7), 1j], ids=["sign", "phase", "quarter"]
)
def test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor(diag_factor):
    """The witness's blind spot, pinned so nobody over-trusts it.

    Any per-port factor of unit modulus — a sign flip from a wave-split
    orientation mistake, or a reference-plane shift — is a diagonal similarity
    ``D S D^-1`` with ``|D_jj| = 1``. It sends ``S21 -> c*S21`` and
    ``S12 -> S12/c`` with ``|c| = 1``, so BOTH magnitudes are untouched and
    reciprocity cannot see it at all.

    This test was written expecting the opposite and failed; the algebra above
    is why. Consequence for issue #489: reciprocity separates amplitude and
    orientation-SCALE defects, but a first gate must NOT be advertised as
    covering orientation sign or reference-plane errors — those need an
    independent handle (a known-phase DUT, or an invariance check).
    """
    n_f = len(_FREQS)
    s_true = _shunt_resistor(25.0, 48.5914, n_f)
    a, b = _plant(s_true, 0.05, n_f)
    good = solve_two_port_from_wave_amplitudes(a, b)
    a_bad, b_bad = a.copy(), b.copy()
    a_bad[1, :] *= diag_factor
    b_bad[1, :] *= diag_factor
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 0]), np.abs(bad.s_params[0, 1]), atol=1e-12
    )
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 0]), np.abs(good.s_params[1, 0]), atol=1e-12
    )



def test_ill_conditioned_drives_warn_and_are_reported():
    """Two nearly identical drives must not silently produce a plausible S.
    The warning must NOT read as "below this threshold is trustworthy" because cond(A) also multiplies the caller's amplitude noise.
    """
    n_f = 2
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    a[0, 0], a[1, 0] = 1.0, 1.0
    a[0, 1], a[1, 1] = 1.0, 1.0 + 1e-12
    b[:, :] = 0.5
    with pytest.warns(UserWarning, match="nearly linearly dependent") as rec:
        out = solve_two_port_from_wave_amplitudes(a, b)
    assert np.all(out.cond_a > 1e3)
    msg = str(rec[0].message).lower()
    assert "degeneracy only" in msg and "not a reliability certificate" in msg


def test_shape_and_finiteness_contract():
    with pytest.raises(ValueError, match="same shape"):
        solve_two_port_from_wave_amplitudes(
            np.zeros((2, 2, 3), dtype=complex), np.zeros((2, 2, 4), dtype=complex)
        )
    with pytest.raises(ValueError, match=r"\(2, 2, n_freqs\)"):
        solve_two_port_from_wave_amplitudes(
            np.zeros((3, 2, 3), dtype=complex), np.zeros((3, 2, 3), dtype=complex)
        )
    a, b = _plant(_through(2), 0.05, 2)
    a[0, 0, 1] = np.nan
    out = solve_two_port_from_wave_amplitudes(a, b)
    assert np.all(np.isfinite(out.s_params[:, :, 0]))
    assert np.all(np.isnan(out.s_params[:, :, 1]))


@pytest.mark.parametrize('axis,coordinate,size,center', [
    ('z', .0006, None, None), ('z', .0594, None, None),
    ('x', .004, (.004, .002), (.004, .001)),
])
def test_extra_flux_monitor_refuses_open_feed_end(axis, coordinate, size, center):
    sim = _sim()
    scratch = _sim()
    scratch.add_flux_monitor(axis=axis, coordinate=coordinate, size=size, center=center, name='open-end')
    with pytest.raises(ValueError, match=r'open (bottom|top) feed end.*run\(\)/forward\(\) port has a shielded end'):
        sim.compute_coaxial_two_port(n_steps=1, freqs=BAND, extra_flux_monitors=scratch._flux_monitors)
