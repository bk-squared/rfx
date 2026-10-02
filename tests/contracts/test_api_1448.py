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
    with pytest.raises(AttributeError, match=r"minimize_s11_at_freq_wave_decomp.*forward\(port_s11_freqs="):
        objectives.minimize_s11_at_freq(1e9)


def test_directivity_refuses_wrong_sign_gradient():
    with pytest.raises(ValueError, match="wrong-sign gradients.*power-changing.*default log_ratio=True"):
        objectives.maximize_directivity(0., 0., log_ratio=False)


def _independent_standard_read(path, n_ports):
    try:
        import skrf
    except ImportError:
        # Minimal RI/Hz Touchstone 1.0 parser, independent of rfx.io helpers.
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
    return skrf.Network(str(path)).s.transpose(1, 2, 0)


@pytest.mark.parametrize("n_ports", [2, 3])
def test_default_touchstone_standard_independent_reader(tmp_path, n_ports):
    values = np.arange(1, 1 + n_ports*n_ports*2).reshape(n_ports, n_ports, 2)
    expected = values / 100 + 1j * (values + 20) / 100
    path = tmp_path / f"nonreciprocal.s{n_ports}p"
    write_touchstone(path, expected, np.array([1e9, 2e9]), version="1.0", fmt="RI", freq_unit="Hz")
    np.testing.assert_allclose(_independent_standard_read(path, n_ports), expected)
    np.testing.assert_allclose(read_touchstone(path)[0], expected)


def test_old_unmarked_legacy_file_explicit_layout(tmp_path):
    path = tmp_path / "old.s3p"
    # Historical column-major fixture, not produced by today's writer.
    path.write_text("# Hz S RI R 50\n1e9 11 0 21 0 31 0 12 0\n 22 0 32 0 13 0 23 0 33 0\n")
    expected = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])[:, :, None]
    np.testing.assert_array_equal(read_touchstone(path, layout="legacy-rfx")[0], expected)
    np.testing.assert_array_equal(read_touchstone_full(path, layout="legacy-rfx").s_params, expected)
    np.testing.assert_array_equal(read_touchstone(path)[0], expected.transpose(1, 0, 2))
    np.testing.assert_array_equal(read_touchstone_full(path).s_params, expected.transpose(1, 0, 2))
    written = tmp_path / "legacy.s3p"
    write_touchstone(written, expected, np.array([1e9]), layout="legacy-rfx")
    np.testing.assert_allclose(read_touchstone(written, layout="legacy-rfx")[0], expected)


def test_adi_default_and_forward_result_export():
    assert inspect.signature(Simulation).parameters["adi_cfl_factor"].default == 2.0
    sim = Simulation(freq_max=1e9, domain=(.03, .03, .03), solver="adi", boundary="pec")
    assert sim._adi_cfl_factor == 2.0
    from rfx.api._spec import ForwardResult
    assert rfx.ForwardResult is ForwardResult
    assert "ForwardResult" in rfx.__all__
