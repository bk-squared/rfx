"""Tests for independent V/I dump replay of port S-matrices.

These tests are algebraic E3-harness checks.  They do not run FDTD and do not
claim additional solver physics validation by themselves.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures"

from rfx import (
    compare_replayed_smatrix,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
    replay_smatrix_from_vi_dump,
)


def _synthetic_two_port_dump():
    freqs = np.array([1.0e9, 2.0e9, 3.0e9])
    z0 = np.array([50.0, 75.0])
    s = np.zeros((2, 2, freqs.size), dtype=np.complex128)
    s[0, 0, :] = [0.10 + 0.02j, 0.08 + 0.01j, 0.05 + 0.00j]
    s[1, 0, :] = [0.85 - 0.03j, 0.82 - 0.04j, 0.78 - 0.06j]
    s[0, 1, :] = [0.83 - 0.02j, 0.80 - 0.03j, 0.76 - 0.04j]
    s[1, 1, :] = [0.07 + 0.01j, 0.06 + 0.02j, 0.04 + 0.01j]

    a = np.zeros((2, 2, freqs.size), dtype=np.complex128)
    b = np.zeros_like(a)
    for driven in range(2):
        incident = (1.0 + 0.1 * driven) * np.exp(1j * np.linspace(0.0, 0.3, freqs.size))
        a[driven, driven, :] = incident
        b[driven, :, :] = s[:, driven, :] * incident.reshape(1, -1)

    sqrt_z = np.sqrt(z0).reshape(1, 2, 1)
    voltages = sqrt_z * (a + b)
    currents = (a - b) / sqrt_z
    # Role-selected dump convention (issue #308): production dumps register
    # the arriving wave at a PASSIVE receive port with the opposite voltage
    # sign relative to the textbook split (the port-cell field sense in
    # V = -E*dx), so the replay's receive channel is b_recv = -(V + Z0*I) /
    # (2*sqrt(Z0)).  Encode that here: flip the voltage sign on off-diagonal
    # (receive) entries; currents already match ((a - b)/sqrt(Z0) with a=0).
    for driven in range(2):
        voltages[driven, 1 - driven, :] *= -1.0
    return freqs, z0, s, voltages, currents


def test_replay_smatrix_from_vi_phasors_matches_known_smatrix():
    freqs, z0, production, voltages, currents = _synthetic_two_port_dump()

    replayed = replay_smatrix_from_vi_dump(
        voltages,
        currents,
        freqs=freqs,
        port_impedances=z0,
        port_names=("in", "out"),
    )

    assert replayed.port_names == ("in", "out")
    np.testing.assert_allclose(replayed.s_params, production, rtol=1e-12, atol=1e-12)
    comparison = compare_replayed_smatrix(
        replayed,
        type("Production", (), {"s_params": production, "freqs": freqs})(),
        atol=1e-12,
        rtol=1e-12,
    )
    assert comparison.ok, comparison.summary()


def test_replay_accepts_positive_out_current_convention():
    freqs, z0, production, voltages, currents = _synthetic_two_port_dump()

    replayed = replay_smatrix_from_vi_dump(
        voltages,
        -currents,
        freqs=freqs,
        port_impedances=z0,
        current_convention="positive_out_of_dut",
    )

    np.testing.assert_allclose(replayed.s_params, production, rtol=1e-12, atol=1e-12)


def test_a_dump_written_before_the_driven_diagonal_still_replays():
    """A lumped dump from the old extractor replays to its own recorded S.

    tests/fixtures/lumped_two_port_vi_dump_pre_driven_diagonal.npz was written
    by the pre-2026-09-21 code on commit b4cf8f29: its production S is the
    per-cell frame — the passive port-branch diagonal and an incident wave
    built on the same (pre-injection) voltage the dump stores — so it carries
    no ``diagonal_frame`` key.

    That absence selects the legacy per-cell frame. Nothing else in the repo
    exercises it with a real stored dump: every other lumped dump test
    generates its dump at run time with today's writer, so a future change
    could break old-dump replay with every test still green.
    """
    path = _FIXTURE_DIR / "lumped_two_port_vi_dump_pre_driven_diagonal.npz"
    dump = load_port_vi_dump_npz(path)

    assert "diagonal_frame" not in dump.metadata, (
        "this fixture is only a legacy-branch witness while it declares no "
        "diagonal frame")
    assert dump.production_smatrix is not None

    replayed = replay_smatrix_from_port_vi_dump(dump)
    comparison = compare_replayed_smatrix(
        replayed,
        type("Production", (), {"s_params": dump.production_smatrix,
                                "freqs": dump.freqs})(),
    )
    assert comparison.ok, comparison.summary()


def test_reference_plane_shift_is_explicit_metadata():
    freqs = np.array([1.0e9])
    z0 = 50.0
    gamma = np.array([1j * 2.0])
    raw_s11 = np.array([[[0.5 + 0.0j]]])
    a_raw = np.array([[[1.0 + 0.0j]]])
    b_raw = raw_s11 * a_raw
    sqrt_z = np.sqrt(z0)
    voltages = sqrt_z * (a_raw + b_raw)
    currents = (a_raw - b_raw) / sqrt_z

    replayed = replay_smatrix_from_vi_dump(
        voltages,
        currents,
        freqs=freqs,
        port_impedances=z0,
        reference_plane_offsets_m=np.array([0.25]),
        propagation_constants=gamma,
    )

    expected = raw_s11 * np.exp(2.0 * gamma[0] * 0.25)
    np.testing.assert_allclose(replayed.s_params, expected, rtol=1e-12, atol=1e-12)
