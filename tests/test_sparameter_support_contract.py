"""Contract tests for port-family-specific S-parameter support.

These tests lock the public API boundary: unsupported port/source families must
fail loudly instead of returning ``None`` after an explicit S-parameter request.
They do not run FDTD.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfx import MSLSMatrixResult, Simulation


MATRIX_PATH = Path("docs/guides/sparameter_support_matrix.json")


def test_sparameter_support_matrix_lists_every_public_port_surface():
    data = json.loads(MATRIX_PATH.read_text())
    primitives = {entry["primitive"] for entry in data["port_families"]}

    expected = {
        "add_port(extent=None)",
        "add_port(extent=...)",
        "add_msl_port(...)",
        "add_waveguide_port(...)",
        "add_coaxial_port(...)",
        "add_floquet_port(...)",
        "add_source(...) / add_polarized_source(...)",
        "add_tfsf_source(...)",
        "add_probe(...) / add_dft_plane_probe(...) / add_flux_monitor(...)",
    }
    assert expected <= primitives
    assert data["result_convention"]["full_s_matrix_shape"] == "(n_ports, n_ports, n_freqs)"
    assert data["result_convention"]["indexing"].startswith("S[receiver_port")


def test_msl_result_is_publicly_importable():
    assert MSLSMatrixResult.__name__ == "MSLSMatrixResult"


def test_run_sparams_rejects_source_only_request():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), boundary="pec")
    sim.add_source((0.005, 0.005, 0.005), "ez")

    with pytest.raises(ValueError, match="add_source.*cannot populate"):
        sim.run(n_steps=1, compute_s_params=True)


def test_run_sparams_rejects_msl_port_family_with_actionable_api():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.006, 0.002), dx=0.5e-3, boundary="pec")
    sim.add_msl_port(
        position=(0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
    )

    with pytest.raises(ValueError, match="compute_msl_s_matrix"):
        sim.run(n_steps=1, compute_s_params=True)


def test_preflight_sparameters_routes_msl_before_running_fdtd():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.006, 0.002), dx=0.5e-3, boundary="pec")
    sim.add_msl_port(
        position=(0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
    )

    issues = sim.preflight_sparameters(calculator="run")

    assert any("compute_msl_s_matrix" in issue for issue in issues)
    assert sim.preflight_sparameters(calculator="msl") == []


def test_run_sparams_rejects_waveguide_port_family_with_actionable_api():
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.023),
        z_range=(0.0, 0.010),
        direction="+x",
        f0=10e9,
        name="wg",
    )

    with pytest.raises(ValueError, match="compute_waveguide_s_matrix"):
        sim.run(n_steps=1, compute_s_params=True)


def test_preflight_sparameters_catches_waveguide_cardinality():
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.023),
        z_range=(0.0, 0.010),
        direction="+x",
        f0=10e9,
        name="wg",
    )

    issues = sim.preflight_sparameters(calculator="waveguide")

    assert any("at least two waveguide ports" in issue for issue in issues)


def test_run_sparams_rejects_floquet_experimental_family():
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.030), dx=1e-3, boundary="cpml")
    sim.add_floquet_port(0.005, axis="z")

    with pytest.raises(ValueError, match="floquet"):
        sim.run(n_steps=1, compute_s_params=True)


def test_run_rejects_unwired_coaxial_port():
    sim = Simulation(freq_max=8e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary="pec")
    sim.add_coaxial_port((0.010, 0.010, 0.015))

    with pytest.raises(NotImplementedError, match="add_coaxial_port"):
        sim.run(n_steps=1, compute_s_params=False)


def test_forward_s11_rejects_non_port_source_only_request():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=1e-3, boundary="pec")
    sim.add_source((0.005, 0.005, 0.005), "ez")

    with pytest.raises(ValueError, match="add_source.*not an impedance port"):
        sim.forward(n_steps=1, port_s11_freqs=np.array([1.0e9]), skip_preflight=True)


def test_forward_tfsf_nonuniform_lane_fenced():
    """Differentiable TFSF forward is wired on the uniform single-device lane
    (see test_forward_tfsf_differentiable.py); the non-uniform lane has no TFSF
    handling and must fail loud rather than silently drop the source."""
    dz = np.array([0.5e-3] * 8)  # dz_profile => non-uniform forward lane
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.004), dx=0.5e-3,
                     dz_profile=dz, boundary="cpml", cpml_layers=6)
    sim.add_tfsf_source(f0=10e9, polarization="ez", direction="+x")
    with pytest.raises(NotImplementedError, match="uniform single-device"):
        sim.forward(n_steps=1, skip_preflight=True)


def test_preflight_sparameters_routes_forward_specialized_families():
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.023),
        z_range=(0.0, 0.010),
        direction="+x",
        f0=10e9,
        name="wg",
    )

    issues = sim.preflight_sparameters(calculator="forward")

    assert any("waveguide ports use compute_waveguide_s_matrix" in issue for issue in issues)


def test_preflight_sparameters_rejects_unknown_calculator():
    sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01), dx=1e-3, boundary="pec")

    with pytest.raises(ValueError, match="Unknown S-parameter calculator"):
        sim.preflight_sparameters(calculator="not-a-calculator")


def _nu_msl_sim(mode: str = "laplace") -> "Simulation":
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.006, 0.002),
        dx=0.5e-3,
        dz_profile=np.full(4, 0.5e-3),
        boundary="pec",
    )
    sim.add_msl_port(
        position=(0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
        mode=mode,
    )
    return sim


def test_compute_msl_s_matrix_nu_laplace_not_fenced():
    """The NU MSL lane is OPEN for the Ez static-Laplace feed (PR feat/msl-nu-runner
    lifted the blanket fence). A laplace port must get PAST the fence — here the
    minimal geometry has no PEC trace, so it proceeds into the extractor and fails
    with the trace-PEC RuntimeError, NOT the old 'uniform Yee lane only'
    NotImplementedError (which would mean the fence still blocks it)."""
    sim = _nu_msl_sim(mode="laplace")
    with pytest.raises(RuntimeError, match="no PEC trace conductor"):
        sim.compute_msl_s_matrix(n_steps=1)


def test_compute_msl_s_matrix_nu_eigenmode_still_fenced():
    """The eigenmode J+M launch stays fenced on the NU lane — run_nonuniform has
    no magnetic-source channel for the Schelkunoff H-source."""
    sim = _nu_msl_sim(mode="eigenmode")
    with pytest.raises(NotImplementedError, match="magnetic-source channel"):
        sim.compute_msl_s_matrix(n_steps=1)


def test_msl_nu_fence_message_parity_sparams_vs_preflight():
    """Contract-test governance: the eigenmode-fence message must be byte-identical
    between compute_msl_s_matrix (_sparams) and preflight_sparameters (_preflight),
    so preflight never green-lights a config the method rejects (or vice versa)."""
    # _sparams raises; _preflight collects the same error into a PreflightReport.
    sim_a = _nu_msl_sim(mode="eigenmode")
    try:
        sim_a.compute_msl_s_matrix(n_steps=1)
        msg_sparams = None
    except NotImplementedError as e:
        msg_sparams = str(e)
    assert msg_sparams is not None, "_sparams must fence eigenmode-on-NU"

    sim_b = _nu_msl_sim(mode="eigenmode")
    report = sim_b.preflight_sparameters(calculator="msl")
    fence_issues = [str(i) for i in report.issues if "magnetic-source channel" in str(i)]
    assert len(fence_issues) == 1, (
        f"_preflight must surface the eigenmode fence exactly once; "
        f"got {[str(i) for i in report.issues]}"
    )
    # The _sparams message must appear VERBATIM in the _preflight issue (the
    # report prefixes it with 'NotImplementedError: '): no fence-message drift.
    assert msg_sparams in fence_issues[0], (
        "fence message drift: _sparams vs _preflight must stay byte-identical.\n"
        f"  _sparams:        {msg_sparams!r}\n  _preflight issue: {fence_issues[0]!r}"
    )


def _nu_waveguide_sim(n_modes: int = 1) -> "Simulation":
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.023, 0.010),
        dx=1e-3,
        dy_profile=np.full(23, 1e-3),
        boundary="cpml",
        cpml_layers=4,
    )
    for x_position, direction, name in ((0.010, "+x", "wg1"), (0.090, "-x", "wg2")):
        sim.add_waveguide_port(
            x_position=x_position,
            y_range=(0.0, 0.023),
            z_range=(0.0, 0.010),
            direction=direction,
            f0=10e9,
            n_modes=n_modes,
            name=name,
        )
    return sim


def test_preflight_waveguide_nu_accepts_flux_route():
    """normalize='flux' is a supported NU waveguide route (the NU flux extractor
    and its AD channel are wired in _sparams; graded-dy Airy fixtures cover it).
    preflight must not red-flag a config compute_waveguide_s_matrix() accepts."""
    report = _nu_waveguide_sim().preflight_sparameters(
        calculator="waveguide", normalize="flux"
    )
    fence_issues = [str(i) for i in report.issues if "non-uniform mesh" in str(i)]
    assert not fence_issues, (
        f"preflight red-flags the supported NU flux route: {fence_issues}"
    )


@pytest.mark.parametrize(
    "normalize, n_modes",
    [(False, 1), (True, 2)],
    ids=["normalize-clause", "multimode-clause"],
)
def test_waveguide_nu_fence_message_parity_sparams_vs_preflight(normalize, n_modes):
    """Same governance as the MSL parity test above: the NU fence message must
    stay byte-identical between compute_waveguide_s_matrix (_sparams) and
    preflight_sparameters (_preflight), clause by clause. Regression lock for
    the drift where preflight kept rejecting normalize='flux' after _sparams
    started accepting it (uniform PR #172 flux-AD mirror on the NU lane)."""
    sim_a = _nu_waveguide_sim(n_modes=n_modes)
    try:
        sim_a.compute_waveguide_s_matrix(n_steps=1, normalize=normalize)
        msg_sparams = None
    except NotImplementedError as e:
        msg_sparams = str(e)
    assert msg_sparams is not None, "_sparams must fence this config on NU"

    sim_b = _nu_waveguide_sim(n_modes=n_modes)
    report = sim_b.preflight_sparameters(calculator="waveguide", normalize=normalize)
    fence_issues = [str(i) for i in report.issues if "non-uniform mesh" in str(i)]
    assert len(fence_issues) == 1, (
        f"_preflight must surface the NU fence exactly once; "
        f"got {[str(i) for i in report.issues]}"
    )
    assert msg_sparams in fence_issues[0], (
        "fence message drift: _sparams vs _preflight must stay byte-identical.\n"
        f"  _sparams:        {msg_sparams!r}\n  _preflight issue: {fence_issues[0]!r}"
    )
