"""Touchstone metadata and physical interop gates."""

from pathlib import Path

import numpy as np
import pytest

from rfx import (
    TouchstoneData,
    read_touchstone,
    read_touchstone_full,
    write_touchstone,
)


def test_read_touchstone_full_parses_v2_s4p_standard_row_wise(tmp_path: Path):
    text = "\n".join([
        "[Version] 2.0",
        "# GHz S RI R 50",
        "[Number of Ports] 4",
        "[Number of Frequencies] 1",
        "[Reference] 50 55 60 65",
        "[Matrix Format] Full",
        "[Network Data]",
        # Standard full matrix order for 3+ ports: row-wise S11,S12,...S44.
        "1.0  1 0 2 0 3 0 4 0",
        "  5 0 6 0 7 0 8 0",
        "  9 0 10 0 11 0 12 0",
        "  13 0 14 0 15 0 16 0",
        "[End]",
    ])
    path = tmp_path / "standard.s4p"
    path.write_text(text)

    data = read_touchstone_full(path)

    assert isinstance(data, TouchstoneData)
    assert data.version == "2.0"
    assert data.layout == "standard"
    np.testing.assert_allclose(data.reference, [50, 55, 60, 65])
    np.testing.assert_allclose(data.freqs, [1e9])
    expected = np.arange(1, 17, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(data.s_params[:, :, 0].real, expected)


def test_legacy_read_touchstone_rejects_nonuniform_references(tmp_path: Path):
    path = tmp_path / "nonuniform.s2p"
    path.write_text("\n".join([
        "[Version] 2.0",
        "# GHz S RI R 50",
        "[Number of Ports] 2",
        "[Number of Frequencies] 1",
        "[Reference] 50 75",
        "[Two-Port Data Order] 12_21",
        "[Network Data]",
        "1.0  11 0 12 0 21 0 22 0",
        "[End]",
    ]))

    data = read_touchstone_full(path)
    np.testing.assert_allclose(data.reference, [50, 75])
    assert data.s_params[0, 1, 0] == 12 + 0j
    assert data.s_params[1, 0, 0] == 21 + 0j

    with pytest.raises(ValueError, match="read_touchstone_full"):
        read_touchstone(path)


def test_v2_uniform_reference_still_supports_legacy_tuple_api(tmp_path: Path):
    s_params = np.zeros((2, 2, 1), dtype=np.complex128)
    s_params[0, 0, 0] = 11
    s_params[0, 1, 0] = 12
    s_params[1, 0, 0] = 21
    s_params[1, 1, 0] = 22
    freqs = np.array([1e9])
    path = tmp_path / "uniform.s2p"

    write_touchstone(
        path,
        s_params,
        freqs,
        version="2.0",
        port_z0=np.array([50.0, 50.0]),
        two_port_order="12_21",
    )

    data = read_touchstone_full(path)
    assert data.two_port_order == "12_21"
    np.testing.assert_allclose(data.s_params, s_params)

    s_read, f_read, z0 = read_touchstone(path)
    np.testing.assert_allclose(s_read, s_params)
    np.testing.assert_allclose(f_read, freqs)
    assert z0 == 50.0


def test_v2_writer_preserves_nonuniform_reference_and_legacy_api_rejects(
    tmp_path: Path,
):
    s_params = np.eye(3, dtype=np.complex128)[:, :, None] * 0.1
    freqs = np.array([1.5e9])
    path = tmp_path / "nonuniform_writer.s3p"

    write_touchstone(
        path,
        s_params,
        freqs,
        version="2.0",
        port_z0=np.array([50.0, 55.0, 60.0]),
    )

    data = read_touchstone_full(path)
    np.testing.assert_allclose(data.s_params, s_params)
    np.testing.assert_allclose(data.reference, [50.0, 55.0, 60.0])
    np.testing.assert_allclose(np.asarray(data.z0), [50.0, 55.0, 60.0])

    with pytest.raises(ValueError, match="read_touchstone_full"):
        read_touchstone(path)


def test_v1_writer_rejects_nonuniform_reference(tmp_path: Path):
    s_params = np.zeros((2, 2, 1), dtype=np.complex128)
    freqs = np.array([1e9])

    with pytest.raises(ValueError, match="non-uniform port_z0"):
        write_touchstone(
            tmp_path / "bad_reference.s2p",
            s_params,
            freqs,
            version="1.0",
            port_z0=np.array([50.0, 75.0]),
        )


def test_option_line_omitted_format_defaults_to_touchstone_ma(tmp_path: Path):
    path = tmp_path / "default_ma.s2p"
    path.write_text("\n".join([
        "# GHz S R 50",
        "1.0  0.5 90  0.1 0  0.1 0  0.5 -90",
    ]))

    data = read_touchstone_full(path)

    assert data.fmt == "MA"
    np.testing.assert_allclose(data.s_params[0, 0, 0], 0.5j, atol=1e-15)
    np.testing.assert_allclose(data.s_params[1, 1, 0], -0.5j, atol=1e-15)
    np.testing.assert_allclose(data.s_params[1, 0, 0], 0.1 + 0j, atol=1e-15)


def test_writer_rejects_extension_port_mismatch_and_nonfinite_data(
    tmp_path: Path,
):
    freqs = np.array([1e9])
    three_port = np.zeros((3, 3, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="extension declares 2 ports"):
        write_touchstone(tmp_path / "bad.s2p", three_port, freqs, version="2.0")

    bad_s = np.zeros((2, 2, 1), dtype=np.complex128)
    bad_s[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_touchstone(tmp_path / "bad_nan.s2p", bad_s, freqs)

    with pytest.raises(ValueError, match="finite"):
        write_touchstone(
            tmp_path / "bad_freq.s2p",
            np.zeros((2, 2, 1), dtype=np.complex128),
            np.array([np.inf]),
        )


def test_reader_accepts_fortran_d_exponents_and_information_block(
    tmp_path: Path,
):
    path = tmp_path / "fortran_info.s2p"
    path.write_text("\n".join([
        "[Version] 2.0",
        "# GHZ S RI R 5.0D+01",
        "[Number of Ports] 2",
        "[Number of Frequencies] 1",
        "[Reference]",
        "5.0D+01 7.5D+01",
        "[Begin Information]",
        "CreatedBy external-rf-tool",
        "Project d-exponent-smoke",
        "[End Information]",
        "[Network Data]",
        "1.0D+00  1.1D+01 0.0D+00 2.1D+01 0.0D+00 1.2D+01 0.0D+00 2.2D+01 0.0D+00",
        "[End]",
    ]))

    data = read_touchstone_full(path)

    np.testing.assert_allclose(data.freqs, [1e9])
    np.testing.assert_allclose(data.reference, [50.0, 75.0])
    assert data.information == (
        "CreatedBy external-rf-tool",
        "Project d-exponent-smoke",
    )
    assert data.s_params[0, 0, 0] == 11 + 0j
    assert data.s_params[1, 0, 0] == 21 + 0j
    assert data.s_params[0, 1, 0] == 12 + 0j
    assert data.s_params[1, 1, 0] == 22 + 0j


def test_v2_writer_roundtrips_information_block(tmp_path: Path):
    s_params = np.eye(2, dtype=np.complex128)[:, :, None] * 0.1
    freqs = np.array([1e9])
    path = tmp_path / "info_writer.s2p"

    write_touchstone(
        path,
        s_params,
        freqs,
        version="2.0",
        information={"Project": "interop", "Tool": "rfx"},
    )

    text = path.read_text()
    assert "[Begin Information]" in text
    assert "[End Information]" in text
    data = read_touchstone_full(path)
    assert data.information == ("Project interop", "Tool rfx")
    np.testing.assert_allclose(data.s_params, s_params)


def test_v1_writer_rejects_information_block(tmp_path: Path):
    with pytest.raises(ValueError, match="information blocks"):
        write_touchstone(
            tmp_path / "bad_info.s1p",
            np.zeros((1, 1, 1), dtype=np.complex128),
            np.array([1e9]),
            information=["Tool rfx"],
        )


def test_v2_writer_defaults_to_standard_row_wise_multiport_layout(tmp_path: Path):
    s_params = np.zeros((3, 3, 1), dtype=np.complex128)
    # Non-symmetric values expose row-major vs column-major ordering bugs.
    for i in range(3):
        for j in range(3):
            s_params[i, j, 0] = 10 * (i + 1) + (j + 1)
    freqs = np.array([2.5e9])
    path = tmp_path / "standard-default.s3p"

    write_touchstone(path, s_params, freqs, version="2.0")
    data = read_touchstone_full(path)

    assert data.version == "2.0"
    assert data.layout == "standard"
    np.testing.assert_allclose(data.s_params, s_params)


def test_legacy_layout_remains_default_for_existing_v1_multiport(tmp_path: Path):
    s_params = np.zeros((3, 3, 1), dtype=np.complex128)
    # Values chosen so row-wise and column-major layouts would differ.
    for i in range(3):
        for j in range(3):
            s_params[i, j, 0] = 10 * (i + 1) + (j + 1)
    freqs = np.array([2e9])
    path = tmp_path / "legacy.s3p"

    write_touchstone(path, s_params, freqs)
    s_read, f_read, z0 = read_touchstone(path)

    np.testing.assert_allclose(s_read, s_params)
    np.testing.assert_allclose(f_read, freqs)
    assert z0 == 50.0


def test_v2_reader_rejects_declared_frequency_count_mismatch(tmp_path: Path):
    path = tmp_path / "bad_count.s2p"
    path.write_text("\n".join([
        "[Version] 2.0",
        "# GHz S RI R 50",
        "[Number of Ports] 2",
        "[Number of Frequencies] 2",
        "[Network Data]",
        "1.0  11 0 21 0 12 0 22 0",
        "[End]",
    ]))

    with pytest.raises(ValueError, match="Number of Frequencies"):
        read_touchstone_full(path)


def test_v2_reader_rejects_port_count_extension_mismatch(tmp_path: Path):
    path = tmp_path / "bad_ports.s2p"
    path.write_text("\n".join([
        "[Version] 2.0",
        "# GHz S RI R 50",
        "[Number of Ports] 3",
        "[Number of Frequencies] 1",
        "[Network Data]",
        "1.0  11 0 21 0 12 0 22 0",
        "[End]",
    ]))

    with pytest.raises(ValueError, match="file extension"):
        read_touchstone_full(path)


def _passivity_excess(s_matrix: np.ndarray) -> float:
    excess = 0.0
    for fi in range(s_matrix.shape[2]):
        mat = s_matrix[:, :, fi]
        max_eval = float(np.linalg.eigvalsh(mat.conj().T @ mat).max())
        excess = max(excess, max_eval - 1.0)
    return max(0.0, excess)


def _reciprocity_error(s_matrix: np.ndarray) -> float:
    return float(np.max(np.abs(s_matrix - np.swapaxes(s_matrix, 0, 1))))


def test_passive_reciprocal_fixture_survives_standard_v2_roundtrip(tmp_path: Path):
    """Physical/interop gate: passive reciprocal network survives Touchstone I/O."""
    n_ports = 4
    freqs = np.array([1e9, 2e9, 3e9])
    s_params = np.zeros((n_ports, n_ports, freqs.size), dtype=np.complex128)
    for fi in range(freqs.size):
        diag = 0.10 + 0.01 * fi
        coupling = 0.025 + 0.005 * fi
        mat = np.full((n_ports, n_ports), coupling, dtype=np.complex128)
        np.fill_diagonal(mat, diag)
        s_params[:, :, fi] = mat

    assert _passivity_excess(s_params) <= 0.0
    assert _reciprocity_error(s_params) == 0.0

    path = tmp_path / "passive_reciprocal.s4p"
    write_touchstone(
        path,
        s_params,
        freqs,
        version="2.0",
        layout="standard",
        port_z0=np.array([50.0, 50.0, 50.0, 50.0]),
        fmt="RI",
    )
    text = path.read_text()
    assert "[Version] 2.0" in text
    assert "[Matrix Format] Full" in text
    assert "[Reference]" in text

    data = read_touchstone_full(path)
    np.testing.assert_allclose(data.s_params, s_params, atol=1e-12)
    np.testing.assert_allclose(data.reference, np.full(n_ports, 50.0))
    assert _passivity_excess(data.s_params) <= 1e-12
    assert _reciprocity_error(data.s_params) <= 1e-12
