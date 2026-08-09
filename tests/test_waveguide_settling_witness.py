"""#538: the energy ring-down settling witness on the waveguide S-matrix path.

The witness is pure host-side post-processing of the ``v_probe_t`` records
the scan already produces for the DFT extraction — nothing is added to the
jitted graph, so S cannot be perturbed; the identity test below pins that
structurally-guaranteed property anyway (a future refactor that moves the
computation run-side would trip it). Fixture is the WR-90-class two-port
straight guide from test_waveguide_geometry_hygiene, deliberately short
records so the truncation warning path is exercised for real.
"""
import warnings

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation

_FREQS = np.linspace(8.2e9, 12.4e9, 5)


def _two_port():
    sim = Simulation(freq_max=float(_FREQS[-1]), domain=(0.12, 0.04, 0.02),
                     dx=0.004, boundary="cpml", cpml_layers=10)
    for x, direction in ((0.02, "+x"), (0.10, "-x")):
        sim.add_waveguide_port(
            x, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(_FREQS), f0=float(np.mean(_FREQS)),
            bandwidth=0.6,
        )
    return sim


def test_settling_populated_and_truncation_warning_fires():
    """All three normalize modes populate settling_db (n_ports,), finite;
    a deliberately short record fires the aggregate truncation warning
    (measured 2026-08-09, worst-of-4-series witness: [-2.3, -1.9] dB at
    num_periods=4.0 — the parameter this test runs — on this
    fixture — far above the -40 dB rule, which is the point)."""
    for mode in (False, True, "flux"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _two_port().compute_waveguide_s_matrix(
                normalize=mode, num_periods=4.0)
        sd = res.settling_db
        assert sd is not None and sd.shape == (2,), (mode, sd)
        assert np.all(np.isfinite(sd)), (mode, sd)
        assert np.all(sd < 0.0), (mode, sd)
        assert any("ringing" in str(w.message) for w in caught), (
            f"normalize={mode!r}: truncation warning did not fire on an "
            f"unsettled record (settling_db={sd})")


def test_longer_record_settles_deeper():
    """Direction sanity: more periods -> more negative witness on the same
    fixture (the falsifier for a witness that reads something other than
    ring-down)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short = _two_port().compute_waveguide_s_matrix(num_periods=4.0)
        long_ = _two_port().compute_waveguide_s_matrix(num_periods=16.0)
    assert float(np.max(long_.settling_db)) < float(np.max(short.settling_db)), (
        short.settling_db, long_.settling_db)


def test_witness_flag_does_not_perturb_s_extractor_level():
    """Direct non-perturbation pair at the extractor level (review round-1
    upgrade over a determinism-only pin): the SAME cfgs list driven with
    return_settling False vs True must return bit-identical S — the flag
    gates only host-side post-processing of records the scan already
    produces. Fixture imitates
    test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity."""
    from rfx.core.yee import init_materials
    from rfx.sources.waveguide_port import (
        WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix,
    )
    # reuse the committed reciprocity fixture's grid helper directly
    # (bare test-module import needs the tests dir on sys.path)
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_simulation import _CompiledWgGrid as _Grid

    a_wg, b_wg, length, dx, nc, f0 = 0.04, 0.02, 0.12, 0.002, 10, 6e9
    grid = _Grid(length, a_wg, b_wg, dx, nc)
    materials = init_materials(grid.shape)
    freqs = jnp.linspace(5.0e9, 7.0e9, 5)
    n_steps = grid.num_timesteps(num_periods=8)

    def _port(x_index, direction):
        return WaveguidePort(
            x_index=x_index, y_slice=(0, grid.ny), z_slice=(0, grid.nz),
            a=(grid.ny - 1) * dx, b=(grid.nz - 1) * dx,
            mode=(1, 0), mode_type="TE", direction=direction,
        )

    cfgs = [
        init_waveguide_port(_port(nc + 5, "+x"), dx, freqs, f0=f0,
                            dft_total_steps=n_steps),
        init_waveguide_port(_port(grid.nx - nc - 6, "-x"), dx, freqs, f0=f0,
                            dft_total_steps=n_steps),
    ]
    s_off = extract_waveguide_s_matrix(
        grid, materials, cfgs, n_steps,
        boundary="cpml", cpml_axes="x", pec_axes="yz",
    )
    s_on, settling = extract_waveguide_s_matrix(
        grid, materials, cfgs, n_steps,
        boundary="cpml", cpml_axes="x", pec_axes="yz", return_settling=True,
    )
    assert np.array_equal(np.asarray(s_off), np.asarray(s_on)), (
        "return_settling=True perturbed S at the extractor level")
    assert settling.shape == (2,) and np.all(np.isfinite(settling))
    assert np.all(settling < 0.0)
