"""``Simulation.compute_coaxial_two_port`` — FDTD path for #489 stage 2.

STATUS: EXPERIMENTAL (see the docstring on ``CoaxialTwoPortResult`` and
``docs/research_notes/20260729_i489_coax_two_port_design.md`` for the full
scope limitation — every gateable DUT is azimuthally symmetric, so this
battery certifies a class that excludes the transition-discontinuity class
issue #489 targets).

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
from rfx.sources.sources import GaussianPulse

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


# ---------------------------------------------------------------------------
# FAST: convention wiring via planted analytic V(z), through the real
# assembly path (no FDTD).
# ---------------------------------------------------------------------------

def _plant_ab(s_true, gamma_t, n_f):
    """Wave amplitudes at both ports for both drives, terminator ``gamma_t``.

    Identical signal-flow construction to
    ``tests/test_coax_two_port_solve.py::_plant`` (same convention:
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


def _voltages_from_ab(a, b, *, gamma, z_planes_m, ref_m, load_below):
    """Invert the extractor's own extrapolation to build V(z) from known a/b.

    ``coaxial_line_reflection_from_plane_voltages`` fits
    ``V(z) = A e^{+gamma z} + B e^{-gamma z}`` and, at ``reference_plane_m``,
    reports ``forward_amp``/``backward_amp`` as one of (A-branch, B-branch)
    depending on ``load_below``. This is the exact algebraic inverse: given
    the desired forward/backward amplitude AT the reference plane, solve for
    (A, B) and evaluate V at every requested z. Used only to PLANT a field;
    the actual extraction (matrix-pencil fit of gamma, then this same
    extrapolation) is exercised for real by
    ``_assemble_coaxial_two_port_from_voltages``.

    ``gamma`` may be a scalar (same propagation constant at every frequency)
    or an array matching ``a``/``b``'s length (a DIFFERENT propagation
    constant per frequency — used to de-degenerate the planted fixture so a
    per-frequency indexing bug in the assembly loop would be caught, not
    just a per-array/per-drive one).
    """
    z_planes_m = np.asarray(z_planes_m, dtype=np.float64)
    a = np.atleast_1d(np.asarray(a, dtype=np.complex128))
    b = np.atleast_1d(np.asarray(b, dtype=np.complex128))
    gamma = np.broadcast_to(np.asarray(gamma, dtype=np.complex128), a.shape)
    z0 = float(z_planes_m.mean())
    zr = ref_m - z0
    # Real extractor: a_wave = A*exp(+gamma*zr) ("travels -z"),
    # b_wave = B*exp(-gamma*zr) ("travels +z"). load_below=True ->
    # forward_amp=a_wave, backward_amp=b_wave; load_below=False ->
    # forward_amp=b_wave, backward_amp=a_wave. Solve for (A, B) given the
    # desired forward_amp=b (physical "out of network"), backward_amp=a
    # (physical "into network").
    if load_below:
        A = b * np.exp(-gamma * zr)   # forward_amp = a_wave = A*exp(gamma*zr) = b (target)
        B = a * np.exp(+gamma * zr)   # backward_amp = b_wave = B*exp(-gamma*zr) = a (target)
    else:
        A = a * np.exp(-gamma * zr)   # backward_amp = a_wave = A*exp(gamma*zr) = a (target)
        B = b * np.exp(+gamma * zr)   # forward_amp = b_wave = B*exp(-gamma*zr) = b (target)
    zc = z_planes_m - z0
    # shape (n_planes, n_freqs); gamma varies per frequency (per column).
    return (
        np.exp(np.multiply.outer(zc, gamma)) * A[None, :]
        + np.exp(np.multiply.outer(zc, -gamma)) * B[None, :]
    )


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
    """
    n_f = 3
    s_true = (
        np.array([0.15 + 0.05j, 0.20 - 0.03j, -0.05 + 0.10j]),
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),   # S12
        np.array([0.75 - 0.20j, 0.60 + 0.30j, 0.55 - 0.40j]),   # S21 == S12 (reciprocal)
        np.array([-0.10 + 0.08j, 0.05 - 0.12j, 0.15 + 0.02j]),
    )
    # Passivity sanity check on the planted truth, per frequency (not
    # required by the algebra under test, but keeps the fixture physically
    # plausible at every one of the 3 distinct frequency points).
    s_mat_per_freq = [
        np.array([[s_true[0][fi], s_true[1][fi]], [s_true[2][fi], s_true[3][fi]]])
        for fi in range(n_f)
    ]
    for s_mat_fi in s_mat_per_freq:
        assert np.all(np.linalg.svd(s_mat_fi, compute_uv=False) <= 1.0)

    gamma_t = 0.05  # each feed's own residual self-mismatch (validated envelope 0.02-0.08)
    a_true, b_true = _plant_ab(s_true, gamma_t, n_f)

    # Lossless synthetic propagation constant, a DIFFERENT beta per
    # frequency (rad/m) — not the same value broadcast across all 3.
    gamma = 1j * np.array([40.0, 50.0, 65.0])
    z_planes_top_m = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    z_planes_bot_m = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ref_top_m = 0.17    # ABOVE probes_top -> load_below=False (port 1, +z end)
    ref_bot_m = -0.02   # BELOW probes_bot -> load_below=True  (port 2, -z end)

    v_top_by_drive = np.stack([
        _voltages_from_ab(
            a_true[0, drive], b_true[0, drive], gamma=gamma,
            z_planes_m=z_planes_top_m, ref_m=ref_top_m, load_below=False,
        )
        for drive in range(2)
    ], axis=0)
    v_bot_by_drive = np.stack([
        _voltages_from_ab(
            a_true[1, drive], b_true[1, drive], gamma=gamma,
            z_planes_m=z_planes_bot_m, ref_m=ref_bot_m, load_below=True,
        )
        for drive in range(2)
    ], axis=0)

    s_params, cond_a, rec_resid, fit_resid, gamma_fit = _assemble_coaxial_two_port_from_voltages(
        z_planes_bot_m=z_planes_bot_m, z_planes_top_m=z_planes_top_m,
        ref_bot_m=ref_bot_m, ref_top_m=ref_top_m,
        v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
    )

    for fi in range(n_f):
        np.testing.assert_allclose(s_params[:, :, fi], s_mat_per_freq[fi], atol=1e-9)
    assert np.all(np.isfinite(rec_resid)) and np.all(rec_resid < 1e-9)
    assert np.all(np.isfinite(fit_resid)) and np.all(fit_resid < 1e-9)
    assert np.all(cond_a < 3.0)
    # The matrix-pencil fit itself recovers the planted (per-frequency)
    # propagation constant exactly on this clean synthetic field (no
    # reference-plane extrapolation involved) — a sanity check on the
    # newly-exposed `gamma` return.
    np.testing.assert_allclose(gamma_fit, np.broadcast_to(gamma, gamma_fit.shape), atol=1e-6)


def test_swapped_ab_convention_fails_the_same_asymmetric_fixture():
    """Non-firing-control's mirror: the WRONG convention must NOT also pass.

    If ``a = forward_amp`` / ``b = backward_amp`` were used instead (the
    convention that holds for the ORIGINAL 1-port method, where the
    reference plane is the DUT, not the feed), the recovered S on this same
    asymmetric fixture must be visibly wrong — otherwise the previous test
    would not actually be discriminating anything.
    """
    n_f = 1
    s_true = (0.15 + 0.05j, 0.75 - 0.20j, 0.75 - 0.20j, -0.10 + 0.08j)
    gamma_t = 0.05
    a_true, b_true = _plant_ab(s_true, gamma_t, n_f)
    gamma = 1j * 50.0
    z_planes_top_m = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    z_planes_bot_m = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ref_top_m, ref_bot_m = 0.17, -0.02

    # Build V(z) with a/b SWAPPED relative to the derived convention.
    v_top_by_drive = np.stack([
        _voltages_from_ab(
            b_true[0, drive], a_true[0, drive], gamma=gamma,
            z_planes_m=z_planes_top_m, ref_m=ref_top_m, load_below=False,
        )
        for drive in range(2)
    ], axis=0)
    v_bot_by_drive = np.stack([
        _voltages_from_ab(
            b_true[1, drive], a_true[1, drive], gamma=gamma,
            z_planes_m=z_planes_bot_m, ref_m=ref_bot_m, load_below=True,
        )
        for drive in range(2)
    ], axis=0)

    s_params, *_ = _assemble_coaxial_two_port_from_voltages(
        z_planes_bot_m=z_planes_bot_m, z_planes_top_m=z_planes_top_m,
        ref_bot_m=ref_bot_m, ref_top_m=ref_top_m,
        v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
    )
    s_mat_true = np.array([[s_true[0], s_true[1]], [s_true[2], s_true[3]]])
    assert not np.allclose(s_params[:, :, 0], s_mat_true, atol=1e-6)


# ---------------------------------------------------------------------------
# FAST: registration/geometry error paths (raise before any FDTD run).
# ---------------------------------------------------------------------------

def test_boundary_must_be_cpml():
    sim = _sim(boundary="pec")
    with pytest.raises(ValueError, match="requires boundary='cpml'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_cpml_axes_must_be_z():
    sim = _sim()
    with pytest.raises(ValueError, match="requires cpml_axes='z'"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1, cpml_axes="xyz")


def test_periodic_axes_rejected():
    sim = _sim()
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("x")
    with pytest.raises(ValueError, match="does not support periodic boundary axes"):
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


def test_adi_solver_rejected():
    sim = _sim(solver="adi")
    with pytest.raises(ValueError, match="supports solver='yee' only"):
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
def test_registered_geometry_rejected(feature):
    sim = _sim()
    shape = Box((0.001, 0.001, 0.010), (0.002, 0.002, 0.011))
    if feature == "geometry":
        sim.add_material("test_dielectric", eps_r=2.0)
        sim.add(shape, material="test_dielectric")
    else:
        sim.add_thin_conductor(shape, sigma_bulk=1.0e4, thickness=35.0e-6)
    with pytest.raises(ValueError, match="constructs the complete line geometry"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


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


@pytest.mark.parametrize("helper", ("matched", "open", "pec_end_cap"))
def test_registered_coax_termination_helper_rejected(helper):
    sim = _sim()
    if helper == "matched":
        sim.add_coaxial_matched_load(target_impedance=50.0)
    elif helper == "open":
        sim.add_coaxial_open_termination()
    else:
        sim.add_coaxial_pec_end_cap()
    with pytest.raises(ValueError, match=r"add_coaxial_\* termination helpers"):
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


# ---------------------------------------------------------------------------
# FAST: layout / fit-check math (raise before any FDTD run).
# ---------------------------------------------------------------------------

def test_domain_too_short_for_two_feed_layout_is_rejected():
    # freq_max=40 GHz still gives dx~0.375mm; a tiny z domain cannot fit both
    # ends' fixed (2/1/3-cell) offsets plus their probe arrays.
    sim = _sim(domain=(0.008, 0.008, 0.006))
    with pytest.raises(ValueError, match="domain too short|probe arrays overlap"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_overlapping_probe_arrays_are_rejected():
    # Enough room for the fixed offsets but not for two 12-plane, 4-cell
    # spaced arrays without touching.
    sim = _sim(domain=(0.008, 0.008, 0.020))
    with pytest.raises(ValueError, match="probe arrays overlap|domain too short"):
        sim.compute_coaxial_two_port(n_steps=1, n_freqs=1)


def test_default_domain_fits_the_default_layout():
    """Non-firing control: the domain this suite's slow test uses must NOT
    trip either fit-check — i.e. the two rejection tests above are testing
    a real constraint, not a constraint that always fires."""
    sim = _sim()  # domain=(0.008, 0.008, 0.060), the slow test's own domain
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


# ---------------------------------------------------------------------------
# FAST: passivity-advisory routing (design-note incidental defect 3 — the
# 1-port result bypasses _finalize_sparam_result; this one must not).
# ---------------------------------------------------------------------------

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
    # Column power 0.05^2 + 5.0^2 >> 1: grossly non-passive.
    s_bad = np.array(
        [[0.05 + 0.0j, 5.0 + 0.0j], [5.0 + 0.0j, 0.05 + 0.0j]]
    )[:, :, None]
    result = _dummy_result(s_bad)
    with pytest.warns(UserWarning, match="passivity"):
        out = _finalize_sparam_result(result, extractor="compute_coaxial_two_port", strict=False)
    assert out is result  # advisory only: the (unreliable) result is still returned


def test_compute_coaxial_two_port_routes_through_finalize_sparam_result():
    """Regression guard for design-note incidental defect 3: greppable proof
    the new method's own source calls the shared passivity-advisory epilogue
    (unlike compute_coaxial_line_reflection, which returns
    CoaxialLineReflectionResult directly)."""
    src = inspect.getsource(Simulation.compute_coaxial_two_port)
    assert "_finalize_sparam_result(" in src


# ---------------------------------------------------------------------------
# SLOW (real FDTD, own process): the full through-line fixture.
#
# The run below is domain=(0.008, 0.008, 0.060), freq_max=40e9 (dx=0.3747mm,
# matching the validated 1-port dx), n_steps=6000, freqs=BAND — captured
# 2026-08-02 via a standalone script (not this test file) run with
# `PYTHONPATH=<this worktree> python3 <script>` (the shared checkout's
# editable install otherwise shadows the worktree —
# feedback_stale_editable_install_shadow). Some gates below are pinned fresh
# from this measurement with margin (|S21|,|S12| >= 0.70; reciprocity
# <1%/<1deg); others are INHERITED numbers reused from the 1-port method or
# the stage-1 table rather than independently derived from this run
# (cond_a < 3.0; recurrence_residual < 0.02; |S11|,|S22| <= 0.08; settling_db
# < -40.0; fit_residual < 0.02 borrows the recurrence gate's number by
# convenience) — see the inline comment on each assertion below for which is
# which. Full measured table:
#
#   f(GHz)  |S11|   |S21|   |S12|   |S22|   |S21|/|S12|  angle(S21)-angle(S12)
#     4.00  0.0092  0.9600  0.9606  0.0086  0.9994        0.0071 deg
#     6.00  0.0086  0.9134  0.9137  0.0086  0.9997        0.0131 deg
#     8.00  0.0054  0.8450  0.8475  0.0060  0.9970        0.0602 deg
#    10.00  0.0144  0.8259  0.8254  0.0104  1.0006        0.2011 deg
#    12.00  0.0502  0.7372  0.7377  0.0379  0.9992        0.0666 deg
#
# status="passed", annulus_cells=3.789, cond_a in [1.037, 1.106],
# recurrence_residual max 0.00297 (well under the 1-port's own 0.02 gate),
# fit_residual max 0.0127, settling_db=[-65.87, -65.59] (well under the
# project's -40 dB ring-down rule), column power (|S_j1|^2+|S_j2|^2) in
# [0.546, 0.923] at all 5 bins (comfortably passive), ZERO warnings emitted
# (no passivity/cond/amplitude advisory fired).
#
# The method does not call self.preflight() (same as the 1-port
# compute_coaxial_line_reflection it mirrors — both build their own
# low-level fixture via rfx.simulation.run directly, outside the
# preflight-covered high-level flow); there is no preflight output to quote.
#
# All 6 qualitative predictions from docs/design_notes/
# i489_stage2_two_port_fdtd_predeclaration.md held: |S21|,|S12| near 1;
# |S11|,|S22| small (within the validated 1-port matched-termination
# envelope, 0.02-0.08); reciprocity ratio near 1 (magnitude AND phase);
# cond(A) small (< the stage-1 table's ~3 figure); recurrence residual
# small; ring-down settled.
#
# MECHANISM CHECK (R5, a POST-HOC consistency check run AFTER this |S21|
# decline was measured — no committed artifact predates it — with a
# disclosed estimator choice made after seeing the own/other split below;
# a lossless through line with |S11|<=0.05 cannot have |S21|=0.74 unless
# the discrete line itself attenuates): the |S21| decline (0.96->0.74,
# 4-12 GHz) equals what the matrix-pencil-fitted Re(gamma) the extractor
# already measures predicts over the reported port separation, not merely
# qualitatively plausible. Per frequency, `|S21_solved| *
# exp(+Re(gamma_bar) * L12)` (L12 = distance between the two reference
# planes, from res.reference_planes; gamma_bar = mean of all 4 available
# fits — both arrays x both drives) landed in [0.979, 0.992] at all 5 band
# points (measured 2026-08-02) — inside the [0.95, 1.05] window this check
# treats as consistent.
#
# WHAT THIS CHECK BINDS AND WHAT IT DOES NOT (do not overclaim it as a
# blanket "not an extraction/referral defect" — it is not that broad): it
# is sensitive to SCALE-type deficits — amplitude mis-normalization, mode
# conversion, a bad wave split — because `gamma` is fit from the field's
# SHAPE along z, not its absolute scale, so a scale bug in |S21| would not
# be echoed by a matching shift in the fitted `gamma`. It is structurally
# BLIND to reference-plane REFERRAL errors: a referral error `delta` at
# either plane scales the wave amplitude by `exp(+/-gamma*delta)` while
# `L12` grows by the same `delta`, so the compensation factor absorbs a
# referral error EXACTLY (independently verified to five decimal places
# even at +30 cells of injected referral error — a wrong reference plane
# passes this check unchanged). So: this check is evidence the |S21|
# deficit is a scale-consistent, shape-explained line attenuation rather
# than an amplitude/mode-conversion/wave-split bug; it says nothing about
# whether either reference plane is correctly placed. The
# under-resolved-annulus recipe (>=4 cells) that this repo already
# documents for REFLECTION accuracy (compute_coaxial_line_reflection)
# evidently applies to TRANSMISSION magnitude here too, even though status
# reports "passed" (annulus_cells only gates under 3.5).
#
# IMPORTANT NUANCE, surfaced rather than smoothed over: the 4 individual
# gamma measurements that go into gamma_bar are NOT mutually consistent —
# "own drive" fits (that array's own source active: top-array during
# top-driven, bot-array during bot-driven) measured Re(gamma) roughly
# 2-8x SMALLER than "other drive" fits (that array receiving the
# transmitted signal), and this split narrows but persists across the whole
# band (measured factor ~8x at 4 GHz, ~2.3x at 12 GHz), even though
# recurrence_residual stays <0.003 for EVERY one of the 4 measurements (the
# two-wave fit itself is clean at each one). The two "own-drive" fits agree
# with each other to ~5%; the two "other-drive" fits agree with each other
# to ~2% — so this is a reproducible, physically-structured split, not
# noise. Using ONLY the "own-drive" pair (the more consistent proxy for
# genuine uniform bulk-line loss, since it is less perturbed by the
# far/receiving feed's own near-field) under-predicts the |S21| decline
# (compensated ratio drifts to ~0.88 at 12 GHz, below the confirm window);
# using ONLY the "other-drive" pair over-predicts it (drifts to ~1.09).
# Averaging all 4 is therefore not an arbitrary smoothing choice — it is
# the reading consistent with an ADDITIVE mechanism: genuine (small)
# uniform bulk attenuation from the discrete annulus, PLUS additional
# loss/scattering concentrated at the RECEIVING feed's own discontinuity
# (which a receiving array's local matrix-pencil fit partially attributes
# to "extra decay" because it sits close to a feed now absorbing the full
# incident power, rather than the weak TFSF-leakage residual it absorbs
# during its own drive). See gamma's field docstring on
# CoaxialTwoPortResult and _assemble_coaxial_two_port_from_voltages for the
# same finding recorded at the API level.
@pytest.mark.slow_physics
def test_matched_through_line_transmits_reciprocally():
    sim = _sim()  # domain=(0.008, 0.008, 0.060), freq_max=40e9
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_coaxial_two_port(n_steps=6000, freqs=BAND)
    assert [str(w.message) for w in caught] == []

    assert res.status == "passed"
    assert res.annulus_cells >= 3.5
    assert np.all(np.isfinite(res.s_params))
    # cond_a < 3.0: INHERITED from the stage-1 table's own measured "~3"
    # figure (rfx.sources.coaxial_port.solve_two_port_from_wave_amplitudes
    # docstring), not pinned fresh from this run (measured max 1.106 here).
    assert np.all(res.cond_a < 3.0)
    # recurrence_residual < 0.02: INHERITED verbatim from the 1-port
    # calibration's own gate (tests/test_coaxial_line_calibration.py),
    # not pinned fresh from this run (measured max 0.00297 here).
    assert np.all(res.recurrence_residual < 0.02)
    # fit_residual < 0.02: the NUMBER is borrowed from the recurrence gate
    # above for convenience (same repo, same 2-decimal round number, ~1.6x
    # margin over the measured max 0.0127) — this is not an independently
    # pinned-from-measurement gate for fit_residual specifically; the
    # 1-port method has no fit_residual gate to inherit from.
    assert np.all(res.fit_residual < 0.02)
    # settling_db < -40.0: INHERITED, the project's general ring-down
    # settling rule (docs/guides/simulation_methodology.md), not pinned
    # from this run (measured [-65.87, -65.59] here).
    assert np.all(res.settling_db < -40.0)

    S = res.s_params
    s11, s21, s12, s22 = S[0, 0], S[1, 0], S[0, 1], S[1, 1]
    # |S21|,|S12| >= 0.70: PINNED fresh from this measurement (min measured
    # 0.7372/0.7377 at 12 GHz, with margin down to 0.70).
    assert np.all(np.abs(s21) >= 0.70), np.abs(s21)
    assert np.all(np.abs(s12) >= 0.70), np.abs(s12)
    # |S11|,|S22| <= 0.08: INHERITED from the validated 1-port
    # matched-termination envelope (tests/test_coaxial_line_calibration.py,
    # 0.02-0.08), not pinned fresh from this run — it happens to coincide
    # with this run's own max (measured 0.0502/0.0379 here) but the number
    # itself is reused, not independently derived from this measurement.
    assert np.all(np.abs(s11) <= 0.08), np.abs(s11)
    assert np.all(np.abs(s22) <= 0.08), np.abs(s22)

    # Reciprocity: both magnitude and phase (a mirror-symmetric through
    # line gives no reason to expect a calibration-scale or orientation
    # defect to hide here).
    assert np.all(np.abs(np.abs(s21) / np.abs(s12) - 1.0) < 0.01)
    phase_diff_deg = np.degrees(np.angle(s21) - np.angle(s12))
    assert np.all(np.abs(phase_diff_deg) < 1.0), phase_diff_deg

    # Mechanism-anchored consistency gate (R5, post-hoc — see the comment
    # above): the |S21| decline must equal what the extractor's own
    # matrix-pencil Re(gamma) fits predict over L12, not merely be asserted
    # plausible. gamma_bar averages all 4 available fits (both arrays x
    # both drives) — see the comment above for why averaging all 4, not a
    # subset, is the physically motivated (disclosed, chosen after seeing
    # the own/other split) reading, and for what this check does and does
    # not bind (scale-sensitive, referral-blind).
    L12 = float(res.reference_planes[0] - res.reference_planes[1])
    gamma_bar = np.mean(res.gamma.real, axis=(0, 1))  # (n_freqs,)
    compensated_s21 = np.abs(s21) * np.exp(gamma_bar * L12)
    assert np.all((compensated_s21 >= 0.95) & (compensated_s21 <= 1.05)), (
        compensated_s21, gamma_bar, L12,
    )

    # Passivity, directly (not just via the advisory that already ran above
    # with zero warnings): column power must stay under 1 for a passive DUT.
    col1 = np.abs(s11) ** 2 + np.abs(s21) ** 2
    col2 = np.abs(s22) ** 2 + np.abs(s12) ** 2
    assert np.all(col1 < 1.0) and np.all(col2 < 1.0)
