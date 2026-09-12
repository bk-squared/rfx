"""Mixed MSL sampling: affine H about a fixed E plane must preserve V/I/S.

Only the field-evolution hook is replaced. The public mixed extractor keeps
its actual power-wave assembly (the existing per-drive ratios, not a newly
substituted BA^-1 solve). These are sampling/dataflow tests, not RF accuracy
or reciprocity tests. The supported envelope here is uniform +/-x, wave
magnitudes, and unprojected S.
"""
from __future__ import annotations

from types import MethodType

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
import rfx.api._sparams as sparams
from rfx.probes.probes import DFTPlaneProbe

U = 2.0**-12
H_GAIN = 1 / 256  # Keeps Zref*I comparable to V; avoids a near-unit S11 oracle.
SLOPE = 0.5 + 0.25j
FREQS = np.array([0.8e9, 1.2e9, 1.6e9])


def _case(direction):
    sim = Simulation(freq_max=20e9, domain=(40 * U, 12 * U, 8 * U),
                     dx=U, cpml_layers=2, boundary="cpml")
    sim.add_material("substrate", eps_r=2.0)
    sim.add(Box((0, 0, 0), (40 * U, 12 * U, 2 * U)), material="substrate")
    sim.add(Box((0, 0, 0), (40 * U, 12 * U, 0)), material="pec")
    sim.add(Box((0, 4 * U, 2 * U), (40 * U, 8 * U, 2 * U)), material="pec")
    feed = (2 if direction == "+x" else 38) * U
    sim.add_msl_port(position=(feed, 6 * U, 0), width=4 * U, height=2 * U,
                     direction=direction, mode="uniform", eps_r_sub=2.0,
                     n_probe_offset=10, n_probe_spacing=2, n_probes=3)
    sim.add_port(position=(20 * U, 6 * U, U), component="ez", impedance=50.0)
    # Independent hand coordinate: feed node 2+10 or 38-10.
    return sim, (12 if direction == "+x" else 28) * U


def _backend(target, direction, mode, captures):
    sign = 1 if direction == "+x" else -1

    def fake_forward(self, grid, materials, debye_spec, lorentz_spec,
                     *, port_s11_freqs, **kwargs):
        del materials, debye_spec, lorentz_spec, kwargs
        freqs = np.asarray(port_s11_freqs)
        n_f = len(freqs)
        yz_shape = (grid.ny, grid.nz)
        y_h = (np.arange(grid.ny) - grid.pad_y_lo + 0.5) * U
        z_h = (np.arange(grid.nz) - grid.pad_z_lo + 0.5) * U
        planes, samples = {}, {"hy": [], "hz": []}
        for entry in self._dft_planes:
            assert entry.axis == "x"
            index = grid.position_to_index((entry.coordinate, 0, 0))[0]
            if entry.component == "ez":
                field = np.ones(yz_shape)  # Production V = sum(Ez*dz) = 2U.
                phase = np.ones(n_f, dtype=complex)
            else:
                # Registration coordinates denote E nodes, but raw H[i]
                # lives at x_i+U/2. No collocation helper enters this oracle.
                x_h = (index - grid.pad_x_lo + 0.5) * U
                samples[entry.component].append(x_h)
                factor = 1 + (SLOPE * (x_h - target) / U if mode["affine"] else 0)
                if entry.component == "hy":
                    profile = np.broadcast_to(z_h[None, :] / U, yz_shape)
                else:
                    assert entry.component == "hz"
                    profile = np.broadcast_to(0.25 * y_h[:, None] / U, yz_shape)
                field = sign * H_GAIN * factor * profile
                phase = np.exp(-1j * 2 * np.pi * freqs * grid.dt / 2)
            accumulator = (jnp.asarray(field, dtype=jnp.complex64)[None, :, :]
                           * jnp.asarray(phase, dtype=jnp.complex64)[:, None, None])
            planes[entry.name] = DFTPlaneProbe(
                accumulator=accumulator, freqs=entry.freqs, component=entry.component,
                axis=0, index=index, total_steps=1, window="rect", window_alpha=0.25,
            )
        msl_driven = self._msl_ports[0].excite
        captures.append((msl_driven, samples))
        # Nonzero lumped incident/receive waves for both drive columns.
        v_lw = (0.2 if msl_driven else -1.0) * 2 * U * np.ones(n_f)
        i_lw = -v_lw / 50.0
        return {"lumped": [(None, (v_lw, i_lw))], "wire": None,
                "dft_planes": planes, "time_series": None}

    return fake_forward


@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_mixed_affine_h_matches_constant_reference_and_refutes_single_plane(
    monkeypatch, direction,
):
    sim, target = _case(direction)
    mode, captures = {"affine": False}, []
    monkeypatch.setattr(sim, "_forward_from_materials",
                        MethodType(_backend(target, direction, mode, captures), sim))

    def extract():
        captures.clear()
        result, raw = sim.compute_mixed_s_matrix(
            freqs=FREQS, n_steps=1, num_periods=1, skip_preflight=True,
            magnitude_channel="wave", enforce_passivity=False, return_diagnostics=True)
        assert raw["drive_plan"] == [("lw", 0), ("msl", 0)]
        assert [driven for driven, _ in captures] == [False, True]
        for _, samples in captures:
            for component in ("hy", "hz"):
                np.testing.assert_array_equal(sorted(samples[component]),
                                              [target - U / 2, target + U / 2])
        assert result.S_raw is None and result.passivity_correction is None
        return result, raw

    reference, ref = extract()
    mode["affine"] = True
    affine, got = extract()
    # Five width-node segments, one normal segment: Hy contributes 5U,
    # Hz contributes -0.25*5U, with the direction sign already oriented.
    expected_current = 0.75 * 5 * U * H_GAIN
    np.testing.assert_allclose(ref["i_msl"], expected_current, rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(got["i_msl"], expected_current, rtol=2e-6, atol=1e-12)
    for key in ("v0_msl", "v_lw", "i_lw"):
        np.testing.assert_array_equal(got[key], ref[key])
    assert np.all(np.abs(ref["i_msl"]) > 0)
    assert np.all(np.abs(reference.S[[0, 1], [1, 0]]) > 1e-2)
    np.testing.assert_allclose(affine.S, reference.S, rtol=2e-6, atol=2e-7)

    # Explicit falsifier: restore only the old right-H sampling, keeping
    # real temporal correction, transverse integration and S assembly.
    def right_only(planes, names, weights):
        del weights
        return tuple(jnp.asarray(planes[right].accumulator) for _, right in names)

    with monkeypatch.context() as mutation:
        mutation.setattr(sparams, "_collocated_msl_h", right_only)
        bypassed, wrong = extract()
    np.testing.assert_allclose(wrong["i_msl"], expected_current * (1 + SLOPE / 2),
                               rtol=2e-6, atol=1e-12)
    voltage = 2 * U
    zi = float(ref["z0_hj_msl"][0]) * expected_current
    expected_s11 = (voltage - zi) / (voltage + zi)
    expected_wrong = (voltage - zi * (1 + SLOPE / 2)) / (voltage + zi * (1 + SLOPE / 2))
    assert abs(expected_wrong - expected_s11) > 0.05
    np.testing.assert_allclose(reference.S[1, 1], expected_s11, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(bypassed.S[1, 1], expected_wrong, rtol=2e-6, atol=2e-7)
    assert np.max(np.abs(bypassed.S - reference.S)) > 0.05
