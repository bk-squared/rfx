"""2.0 API decisions: refusals and independent Touchstone interoperability."""
import inspect

import numpy as np
import pytest

import rfx
from rfx import Simulation
from rfx import optimize_objectives as objectives
from rfx.io import read_touchstone, read_touchstone_full, write_touchstone


def test_removed_periodic_axes_names_replacement():
    sim = Simulation(freq_max=1e9, domain=(.03, .03, .03))
    with pytest.raises(AttributeError, match="BoundarySpec.*per-face periodic"):
        sim.set_periodic_axes("x")
    assert "set_periodic_axes" not in Simulation.__dict__


@pytest.mark.parametrize("faces", [None, {"z_lo"}])
def test_removed_pec_faces_names_replacement(faces):
    with pytest.raises(TypeError, match="BoundarySpec PEC faces"):
        Simulation(freq_max=1e9, domain=(.03, .03, .03), pec_faces=faces)
    assert "pec_faces" not in inspect.signature(Simulation).parameters


def test_removed_time_gated_objective_names_replacement():
    from rfx.optimize_objectives import minimize_s11_at_freq
    assert callable(minimize_s11_at_freq)
    with pytest.raises(AttributeError, match=r"minimize_s11_at_freq_wave_decomp.*forward\(port_s11_freqs="):
        minimize_s11_at_freq(1e9)


def test_directivity_refuses_wrong_sign_gradient():
    with pytest.raises(ValueError, match="wrong-sign gradients.*power-changing.*default log_ratio=True"):
        objectives.maximize_directivity(0., 0., log_ratio=False)


def _independent_standard_read(path, n_ports):
    # Minimal RI/Hz flat parser, independent of rfx.io and scikit-rf.
    lines = path.read_text().splitlines()
    assert "# Hz S RI R 50" in lines
    tokens = [float(t) for line in lines if line and not line.startswith(("!", "#"))
              for t in line.split()]
    records = np.array(tokens).reshape(-1, 1 + 2 * n_ports**2)
    pairs = records[:, 1::2] + 1j * records[:, 2::2]
    matrices = pairs.reshape(-1, n_ports, n_ports)
    if n_ports == 2:
        matrices = matrices.transpose(0, 2, 1)
    return matrices.transpose(1, 2, 0)


@pytest.mark.parametrize("n_ports", [2, 3, 4, 5])
def test_default_touchstone_standard_independent_reader(tmp_path, n_ports):
    values = np.arange(1, 1 + n_ports*n_ports*2).reshape(n_ports, n_ports, 2)
    expected = values / 100 + 1j * (values + 20) / 100
    path = tmp_path / f"nonreciprocal.s{n_ports}p"
    write_touchstone(path, expected, np.array([1e9, 2e9]), version="1.0", fmt="RI", freq_unit="Hz")
    assert "! rfx Touchstone export\n! rfx layout: standard\n" in path.read_text()
    lines = [line.split() for line in path.read_text().splitlines()
             if line and not line.startswith(("!", "#"))]
    lengths = [len(line) for line in lines]
    row_lengths = [2 * min(4, n_ports - start) for start in range(0, n_ports, 4)]
    expected_lengths = row_lengths * n_ports if n_ports > 2 else [2 * n_ports**2]
    expected_lengths[0] += 1
    assert lengths == expected_lengths * 2
    np.testing.assert_allclose(_independent_standard_read(path, n_ports), expected)
    try:
        import skrf
    except ImportError:
        pass  # Optional extra interop check; the independent parser always runs.
    else:
        np.testing.assert_allclose(skrf.Network(str(path)).s.transpose(1, 2, 0), expected)
    np.testing.assert_allclose(read_touchstone_full(path).s_params, expected)
    np.testing.assert_allclose(read_touchstone(path)[0], expected)


@pytest.mark.parametrize("n_ports", [3, 4])
def test_old_writer_default_layout(tmp_path, n_ports):
    # Frozen bytes from origin/main's writer, including its identifying header.
    from pathlib import Path
    path = Path(__file__).parents[1] / "fixtures" / "touchstone" / f"old-rfx.s{n_ports}p"
    values = np.arange(1, 1 + n_ports*n_ports*2).reshape(n_ports, n_ports, 2)
    expected = values / 100 + 1j * (values + 20) / 100
    np.testing.assert_allclose(read_touchstone(path)[0], expected)
    data = read_touchstone_full(path)
    assert data.layout == "legacy-rfx"
    np.testing.assert_allclose(data.s_params, expected)
    np.testing.assert_allclose(read_touchstone(path, layout="standard")[0], expected.transpose(1, 0, 2))
    external = tmp_path / f"external.s{n_ports}p"
    external.write_text(path.read_text().replace("! rfx Touchstone export\n", ""))
    np.testing.assert_allclose(read_touchstone(external)[0], expected.transpose(1, 0, 2))
    np.testing.assert_allclose(read_touchstone_full(external).s_params, expected.transpose(1, 0, 2))


def test_adi_default_and_forward_result_export():
    assert inspect.signature(Simulation).parameters["adi_cfl_factor"].default == 2.0
    sim = Simulation(freq_max=1e9, domain=(.03, .03, .03), solver="adi", boundary="pec")
    assert sim._adi_cfl_factor == 2.0
    from rfx.api._spec import ForwardResult
    assert rfx.ForwardResult is ForwardResult
    assert "ForwardResult" in rfx.__all__


@pytest.mark.parametrize("version,n_ports", [("1.0", 2), ("2.0", 2), ("2.0", 3)])
def test_unmarked_header_does_not_change_two_port_or_v2(tmp_path, version, n_ports):
    values = np.arange(n_ports**2).reshape(n_ports, n_ports, 1) + 1j
    path = tmp_path / f"unmarked.s{n_ports}p"
    write_touchstone(path, values, np.array([1e9]), version=version)
    path.write_text(path.read_text().replace("! rfx layout: standard\n", ""))
    np.testing.assert_allclose(read_touchstone(path)[0], values)
    np.testing.assert_allclose(read_touchstone_full(path).s_params, values)


def test_explicit_legacy_layout_overrides_marker(tmp_path):
    path = tmp_path / "legacy.s3p"
    values = np.arange(9).reshape(3, 3, 1) + 1j
    write_touchstone(path, values, np.array([1e9]), layout="legacy-rfx")
    assert "! rfx layout: legacy-rfx" in path.read_text()
    np.testing.assert_allclose(read_touchstone(path, layout="legacy-rfx")[0], values)
    np.testing.assert_allclose(read_touchstone_full(path, layout="legacy-rfx").s_params, values)


@pytest.mark.parametrize("n_ports", [3, 4, 5])
def test_default_read_follows_the_marker_value(tmp_path, n_ports):
    # A file written today with layout="legacy-rfx" carries that marker; the
    # default reader must honour its value, not just its presence (#1448 review).
    values = (np.arange(n_ports * n_ports).reshape(n_ports, n_ports, 1) + 1
              + 1j * (np.arange(n_ports * n_ports).reshape(n_ports, n_ports, 1) + 50))
    path = tmp_path / f"legacy_marked.s{n_ports}p"
    write_touchstone(path, values, np.array([1e9]), layout="legacy-rfx")
    assert "! rfx layout: legacy-rfx" in path.read_text()
    np.testing.assert_allclose(read_touchstone(path)[0], values)
    np.testing.assert_allclose(read_touchstone_full(path).s_params, values)
