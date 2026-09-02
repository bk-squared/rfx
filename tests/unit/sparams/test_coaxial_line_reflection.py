"""Coaxial transmission-line reflection method (broad-E5 redesign): the
matrix-pencil extractor and its end-to-end short/open/matched calibration.

One file for the field / impedance side of the coax lane (tier 3b of the
2026-09 test-corpus reorganisation, see
``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``). Sections,
each formerly its own file:

1. End-to-end calibration of ``compute_coaxial_line_reflection`` — was
   ``test_coaxial_line_calibration.py``. On a real coax line with a matched
   CPML feed and a >=~4-cell annulus, the canonical terminations must hit
   their analytic targets across the band::

       short   -> Gamma = -1   (|S11| ~ 1, angle ~ 180 deg)
       open    -> Gamma = +1   (|S11| ~ 1)
       matched -> Gamma ~ 0     (|S11| small), and the inferred numerical Z0
                  matches the analytic Z_TEM.

   These are the validated-envelope targets (short/open |Gamma|=1.00-1.03,
   matched 0.02-0.05 at dx=0.375mm); the tolerances reflect that envelope and
   are NOT loosened. The method also flags an under-resolved annulus. The FDTD
   runs are marked ``slow_physics`` (deselected by default); the registration
   / geometry rejection paths are fast.

2. Fast analytic unit tests for the extractor
   ``coaxial_line_reflection_from_plane_voltages`` — was
   ``test_coaxial_line_extraction.py``. No FDTD: feed the matrix-pencil
   estimator synthetic two-wave modal voltages with a KNOWN propagation
   constant and reflection, and assert it recovers them exactly; plus the
   AD-traceability legs of the jnp core.

The ``CoaxialPort`` primitive and the TEM reference-plane V/I helpers stay in
``tests/unit/ports/test_coaxial_port.py`` (port primitive → ports, tier-4b
rule). Every assertion, tolerance, fixture value and parametrisation of the
absorbed files is kept verbatim.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Box
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import (
    coaxial_tem_characteristic_impedance, SMA_PIN_RADIUS, SMA_OUTER_RADIUS,
    coaxial_line_reflection_from_plane_voltages as extract,
)


# ===========================================================================
# formerly tests/unit/sparams/test_coaxial_line_reflection.py
# ===========================================================================

BAND = jnp.asarray([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def _run(termination, freq_max=40.0e9, n_steps=5000, **kwargs):
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=freq_max, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim.compute_coaxial_line_reflection(
        termination=termination, n_steps=n_steps, freqs=BAND, **kwargs)


@pytest.mark.slow_physics
def test_short_reflects_minus_one_full_band():
    res = _run("short")
    assert res.status == "passed"
    assert res.annulus_cells >= 3.5
    mag = np.abs(res.s11)
    # lossless short: |Gamma| = 1 across the band (validated 1.00-1.03)
    assert np.all(np.abs(mag - 1.0) < 0.05), mag
    # phase near +-180 deg (Gamma = -1): cos(angle) strongly negative
    assert np.all(np.cos(np.angle(res.s11)) < -0.85), np.degrees(np.angle(res.s11))
    assert np.all(res.recurrence_residual < 0.02), res.recurrence_residual


@pytest.mark.slow_physics
def test_open_reflects_unity_magnitude_full_band():
    res = _run("open")
    assert res.status == "passed"
    mag = np.abs(res.s11)
    assert np.all(np.abs(mag - 1.0) < 0.05), mag
    assert np.all(res.recurrence_residual < 0.02), res.recurrence_residual


@pytest.mark.slow_physics
def test_matched_reflects_near_zero_and_recovers_z0():
    res = _run("matched")
    assert res.status == "passed"
    mag = np.abs(res.s11)
    # matched load -> |Gamma| small (validated 0.02-0.05)
    assert np.all(mag < 0.08), mag
    # inferred numerical Z0 matches analytic Z_TEM within 15%
    z0_an = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)
    z0_num = np.real(res.z0_numerical_ohm)
    assert np.all(np.abs(z0_num - z0_an) / z0_an < 0.15), (z0_num, z0_an)


@pytest.mark.slow_physics
def test_resistive_load_reflection_magnitude():
    # known mismatch R=25 ohm on the 48.6 ohm SMA line:
    # |Gamma| = |(25 - 48.6)/(25 + 48.6)| = 0.321 (exact analytic, non-trivial).
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    res = sim.compute_coaxial_line_reflection(
        termination="matched", dut_impedance=25.0, n_steps=5000, freqs=BAND)
    assert res.status == "passed"
    z0 = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)
    g_an = abs((25.0 - z0) / (25.0 + z0))
    assert np.all(np.abs(np.abs(res.s11) - g_an) < 0.05), (np.abs(res.s11), g_an)


@pytest.mark.slow_physics
def test_under_resolved_annulus_is_flagged():
    # freq_max=20 GHz -> dx~0.75 mm -> ~1.9-cell annulus (below the >=4 recipe).
    # At this dx the z domain fits 9 of the default 12 probe planes. Until
    # af167a9 the extractor silently dropped the surplus planes, so this test
    # always measured on 9; the fail-loud layout check now requires the count
    # to be requested explicitly. The observable under test (the under_resolved
    # flag) is independent of the probe count.
    res = _run("short", freq_max=20.0e9, n_steps=1500, probe_count=9)
    assert res.annulus_cells < 3.5
    assert res.status == "under_resolved"


@pytest.mark.parametrize(
    "profile_kw",
    [
        {"dx_profile": np.full(8, 1.0e-3)},
        {"dy_profile": np.full(8, 1.0e-3)},
        {"dz_profile": np.full(40, 1.0e-3)},
    ],
    ids=("dx_profile", "dy_profile", "dz_profile"),
)
def test_nonuniform_profiles_are_rejected_before_coaxial_line_run(profile_kw):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
        **profile_kw,
    )
    sim.add_coaxial_port(
        (0.004, 0.004, 0.020),
        face="top",
        pin_length=5.0e-3,
        waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2),
    )

    with pytest.raises(ValueError, match="only a uniform Yee grid"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_existing_tfsf_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    sim.add_tfsf_source(f0=8.0e9)

    with pytest.raises(ValueError, match="existing TFSF source"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_refinement_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    sim.add_refinement((0.018, 0.022), ratio=2, validation="research")

    with pytest.raises(ValueError, match="does not support SBP-SAT refinement"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_adi_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
        solver="adi",
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="supports solver='yee' only"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize(
    "boundary_kw",
    [
        {"boundary": "pec"},
        {"boundary": "upml"},
        {"boundary": "cpml", "cpml_layers": 0},
    ],
    ids=("pec", "upml", "zero_cpml_layers"),
)
def test_nonabsorbing_boundary_is_rejected_before_coaxial_line_run(boundary_kw):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        dx=1.0e-3,
        **boundary_kw,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="requires boundary='cpml'"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_two_dimensional_mode_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
        mode="2d_tmz",
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="requires mode='3d'"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_fourth_order_stencil_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
        stencil_order=4,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="requires stencil_order=2"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_mixed_precision_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
        precision="mixed",
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="requires precision='float32'"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("cpml_axes", ("", "x", "xyz"))
def test_non_axial_cpml_selection_is_rejected_before_coaxial_line_run(cpml_axes):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="requires cpml_axes='z'"):
        sim.compute_coaxial_line_reflection(
            n_steps=1,
            n_freqs=1,
            cpml_axes=cpml_axes,
        )


def test_periodic_axis_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("x")

    with pytest.raises(ValueError, match="does not support periodic boundary axes"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize(
    "z_boundary",
    [
        {"lo": "pec", "hi": "cpml"},
        {"lo": "cpml", "hi": "pec"},
        {"lo": "cpml", "hi": "cpml", "lo_thickness": 0},
        {"lo": "cpml", "hi": "cpml", "hi_thickness": 0},
    ],
    ids=("z_lo_pec", "z_hi_pec", "z_lo_zero", "z_hi_zero"),
)
def test_nonabsorbing_z_face_is_rejected_before_coaxial_line_run(z_boundary):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary={"x": "pec", "y": "pec", "z": z_boundary},
        cpml_layers=16,
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="positive CPML thickness on both z faces"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize(
    "boundary",
    [
        {"x": {"lo": "pec", "hi": "cpml"}, "y": "cpml", "z": "cpml"},
        {"x": {"lo": "pmc", "hi": "pmc"}, "y": "cpml", "z": "cpml"},
        {"x": "cpml", "y": {"lo": "cpml", "hi": "pec"}, "z": "cpml"},
    ],
    ids=("x_pec", "x_pmc", "y_pec"),
)
def test_mixed_transverse_boundary_is_rejected_before_coaxial_line_run(boundary):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary=boundary,
        cpml_layers=16,
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="CPML tokens on all six boundary faces"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("feature", ("geometry", "thin_conductor"))
def test_registered_geometry_is_rejected_before_coaxial_line_run(feature):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    shape = Box((0.001, 0.001, 0.010), (0.002, 0.002, 0.011))
    if feature == "geometry":
        sim.add_material("test_dielectric", eps_r=2.0)
        sim.add(shape, material="test_dielectric")
    else:
        sim.add_thin_conductor(shape, sigma_bulk=1.0e4, thickness=35.0e-6)

    with pytest.raises(ValueError, match="constructs the complete line geometry"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


def test_lumped_rlc_is_rejected_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    sim.add_lumped_rlc((0.004, 0.004, 0.010), R=50.0, topology="parallel")

    with pytest.raises(ValueError, match="registered lumped RLC elements"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("monitor", ("probe", "dft", "flux", "ntff"))
def test_registered_monitor_is_rejected_before_coaxial_line_run(monitor):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    if monitor == "probe":
        sim.add_probe((0.004, 0.004, 0.020))
    elif monitor == "dft":
        sim.add_dft_plane_probe(axis="z", coordinate=0.020, n_freqs=1)
    elif monitor == "flux":
        sim.add_flux_monitor(axis="z", coordinate=0.020, n_freqs=1)
    else:
        sim.add_ntff_box((0.001, 0.001, 0.010), (0.007, 0.007, 0.030), n_freqs=1)

    with pytest.raises(ValueError, match="does not consume registered"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("helper", ("matched", "open", "pec_end_cap"))
def test_registered_coax_termination_helper_is_rejected_before_line_run(helper):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")
    if helper == "matched":
        sim.add_coaxial_matched_load(target_impedance=50.0)
    elif helper == "open":
        sim.add_coaxial_open_termination()
    else:
        sim.add_coaxial_pec_end_cap()

    with pytest.raises(ValueError, match=r"add_coaxial_\* termination helpers"):
        sim.compute_coaxial_line_reflection(n_steps=1, n_freqs=1)


@pytest.mark.parametrize("termination", ("short", "open"))
def test_dut_impedance_is_rejected_when_termination_does_not_use_it(termination):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="used only with termination='matched'"):
        sim.compute_coaxial_line_reflection(
            termination=termination,
            dut_impedance=75.0,
            n_steps=1,
            n_freqs=1,
        )


def test_all_requested_probe_planes_must_fit_before_coaxial_line_run():
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="of 100 requested probe planes fit"):
        sim.compute_coaxial_line_reflection(
            n_steps=1,
            n_freqs=1,
            probe_count=100,
        )


@pytest.mark.parametrize("probe_count", (0, 1, 2))
def test_at_least_three_probe_planes_are_required_before_coaxial_line_run(
    probe_count,
):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="probe_count must be at least 3"):
        sim.compute_coaxial_line_reflection(
            n_steps=1,
            n_freqs=1,
            probe_count=probe_count,
        )


@pytest.mark.parametrize("probe_count", (True, 3.5))
def test_probe_count_must_be_an_integer_before_coaxial_line_run(probe_count):
    sim = Simulation(
        domain=(0.008, 0.008, 0.040),
        freq_max=40.0e9,
        boundary="cpml",
        dx=1.0e-3,
    )
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top")

    with pytest.raises(ValueError, match="probe_count must be an integer"):
        sim.compute_coaxial_line_reflection(
            n_steps=1,
            n_freqs=1,
            probe_count=probe_count,
        )


# ===========================================================================
# formerly tests/unit/sparams/test_coaxial_line_reflection.py
# ===========================================================================

def _synth(beta, gamma_load, *, ref_m, planes, A=1.0, alpha=0.0):
    """V(z)=A e^{+γz}+B e^{-γz} with γ=α+jβ, tuned so Γ(ref_m)=gamma_load
    when the load is BELOW the probe span."""
    g = alpha + 1j * beta
    # incident (toward -z) = A e^{+γz}; Γ = (B e^{-γ ref})/(A e^{+γ ref})
    B = gamma_load * A * np.exp(+2.0 * g * ref_m)
    z = np.asarray(planes, float)
    V = A * np.exp(+g * z) + B * np.exp(-g * z)
    return z, V


@pytest.mark.parametrize("gamma_load", [-1.0 + 0j, 1.0 + 0j, 0.3 + 0.4j, -0.2 - 0.5j, 0.0 + 0j])
def test_recovers_known_reflection_lossless(gamma_load):
    beta = 180.0  # rad/m
    ref = 0.000
    planes = ref + np.array([6, 10, 14, 18, 22, 26]) * 0.75e-3  # equally spaced, above load
    z, V = _synth(beta, gamma_load, ref_m=ref, planes=planes)
    out = extract(z, V, reference_plane_m=ref)
    assert out.reflection == pytest.approx(gamma_load, abs=1e-6)
    assert np.imag(out.gamma) == pytest.approx(beta, rel=1e-6)
    assert abs(np.real(out.gamma)) < 1e-6          # lossless => alpha ~ 0
    assert out.recurrence_residual < 1e-9
    assert out.fit_residual < 1e-9


def test_recovers_known_reflection_with_loss():
    beta, alpha, gamma_load = 240.0, 12.0, 0.5 - 0.3j
    ref = 0.0
    planes = ref + np.array([5, 9, 13, 17, 21]) * 0.6e-3
    z, V = _synth(beta, gamma_load, ref_m=ref, planes=planes, alpha=alpha)
    out = extract(z, V, reference_plane_m=ref)
    assert out.reflection == pytest.approx(gamma_load, abs=1e-6)
    assert np.imag(out.gamma) == pytest.approx(beta, rel=1e-6)
    assert np.real(out.gamma) == pytest.approx(alpha, rel=1e-6)
    assert out.recurrence_residual < 1e-9


def test_load_above_probes_branch():
    """Reference plane ABOVE the probe span: incident wave is +z (the B term)."""
    beta, gamma_load = 200.0, 0.6 + 0.0j
    g = 1j * beta
    ref = 0.030
    planes = np.array([6, 10, 14, 18, 22]) * 0.75e-3  # below ref
    # incident toward +z = B e^{-γz}; Γ = (A e^{+γ ref})/(B e^{-γ ref})
    B = 1.0
    A = gamma_load * B * np.exp(-2.0 * g * ref)
    z = planes
    V = A * np.exp(+g * z) + B * np.exp(-g * z)
    out = extract(z, V, reference_plane_m=ref)
    assert out.reflection == pytest.approx(gamma_load, abs=1e-6)
    assert abs(out.reflection) == pytest.approx(0.6, abs=1e-6)


def test_lossless_reflection_magnitude_is_unity_for_reactive_load():
    """A purely reactive (|Γ|=1) load is recovered with |Γ|=1 regardless of phase
    — the property that the single-plane V/I path violated (|S11|>1)."""
    beta = 150.0
    ref = 0.0
    planes = ref + np.arange(6) * 4 * 0.5e-3 + 0.004
    for phase_deg in (179, 120, 90, 30):
        gamma_load = np.exp(1j * np.radians(phase_deg))
        z, V = _synth(beta, gamma_load, ref_m=ref, planes=planes)
        out = extract(z, V, reference_plane_m=ref)
        assert abs(out.reflection) == pytest.approx(1.0, abs=1e-6)


def test_input_validation():
    z = np.array([1.0, 2.0, 3.0]) * 1e-3
    V = np.array([1 + 0j, 0.5 + 0j, 0.2 + 0j])
    # too few planes
    with pytest.raises(ValueError):
        extract(z[:2], V[:2], reference_plane_m=0.0)
    # unequal spacing
    with pytest.raises(ValueError):
        extract(np.array([1.0, 2.0, 4.0]) * 1e-3, V, reference_plane_m=0.0)
    # all-zero voltages (concrete path keeps the informative raise)
    with pytest.raises(ValueError):
        extract(z, np.zeros(3, complex), reference_plane_m=0.0)


# --- AD-traceability (coax AD-traceable extractor) --------------------------
# Traced voltages dispatch to a jax.numpy core; concrete voltages keep the
# byte-identical NumPy path. These gate the differentiable path.

def test_reflection_extractor_grad_matches_closed_form_and_fd():
    """On a planted two-wave profile V = θ·e^{+jβz} + 0.3·e^{-jβz} the load
    reflection is Γ = 0.3/θ, so d|Γ|/dθ = -0.3/θ². Gate the AD gradient against
    BOTH the closed form and a central finite difference."""
    z = np.linspace(0.002, 0.013, 12)

    def obj(theta):
        V = theta * jnp.exp(1j * 300.0 * jnp.asarray(z)) + 0.3 * jnp.exp(
            -1j * 300.0 * jnp.asarray(z)
        )
        return jnp.abs(extract(z, V, reference_plane_m=0.0).reflection)

    g = float(jax.grad(obj)(jnp.asarray(0.7)))
    assert np.isfinite(g), f"gradient not finite: {g}"
    closed_form = -0.3 / 0.7 ** 2
    assert abs(g - closed_form) < 1e-4, f"AD {g:.8f} vs closed form {closed_form:.8f}"

    h = 1e-4
    fd = (float(obj(0.7 + h)) - float(obj(0.7 - h))) / (2 * h)
    assert abs(g - fd) / max(abs(fd), 1e-12) < 1e-3, f"AD {g:.8f} vs FD {fd:.8f}"


def test_reflection_grad_finite_at_reactive_null():
    """Double-``where`` robustness: a purely reactive |Γ|=1 load — the match/null
    regime that would leak 0·inf=nan through a naive sqrt/divide — keeps a finite
    gradient."""
    z = np.linspace(0.002, 0.013, 12)

    def obj(theta):
        # B on the unit circle, A=θ  ->  |Γ| = 1/θ  (=1 at θ=1)
        V = theta * jnp.exp(1j * 150.0 * jnp.asarray(z)) + np.exp(
            1j * np.radians(120.0)
        ) * jnp.exp(-1j * 150.0 * jnp.asarray(z))
        return jnp.abs(extract(z, V, reference_plane_m=0.0).reflection)

    val = float(obj(jnp.asarray(1.0)))
    g = float(jax.grad(obj)(jnp.asarray(1.0)))
    assert abs(val - 1.0) < 1e-3, f"|Gamma| not unity: {val}"
    assert np.isfinite(g), f"gradient not finite at reactive null: {g}"
