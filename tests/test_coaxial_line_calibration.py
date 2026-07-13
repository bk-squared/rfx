"""End-to-end calibration of the coaxial transmission-line reflection method
(broad-E5 redesign). On a real coax line with a matched CPML feed and a
≥~4-cell annulus, the canonical terminations must hit their analytic targets
across the band:

    short   -> Gamma = -1   (|S11| ~ 1, angle ~ 180 deg)
    open    -> Gamma = +1   (|S11| ~ 1)
    matched -> Gamma ~ 0     (|S11| small), and the inferred numerical Z0
               matches the analytic Z_TEM.

These are the validated-envelope targets (short/open |Gamma|=1.00-1.03,
matched 0.02-0.05 at dx=0.375mm); the tolerances reflect that envelope and are
NOT loosened. The method also flags an under-resolved annulus.

Marked slow_physics (FDTD runs); deselected by default.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Box
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import (
    coaxial_tem_characteristic_impedance, SMA_PIN_RADIUS, SMA_OUTER_RADIUS,
)

BAND = jnp.asarray([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def _run(termination, freq_max=40.0e9, n_steps=5000):
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=freq_max, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim.compute_coaxial_line_reflection(
        termination=termination, n_steps=n_steps, freqs=BAND)


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
    res = _run("short", freq_max=20.0e9, n_steps=1500)
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
