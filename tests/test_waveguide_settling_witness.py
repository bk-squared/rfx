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
    (measured 2026-08-07: [-5.8, -1.9] dB at num_periods=6 on this
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


def test_witness_does_not_perturb_s():
    """Non-perturbation, pinned as run-to-run determinism of the public
    path plus the structural argument: return_settling only gates
    HOST-SIDE post-processing of the ``v_probe_t`` records the scan
    already produces (see settling_db_from_port_records — nothing enters
    the jitted graph), so S cannot differ with the flag; two identical
    public-path runs must therefore agree bit-for-bit, and any future
    refactor that moves the witness run-side breaks either this
    determinism pin or the suite's existing waveguide S value gates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = _two_port().compute_waveguide_s_matrix(num_periods=4.0)
        r2 = _two_port().compute_waveguide_s_matrix(num_periods=4.0)
    assert np.array_equal(np.asarray(r1.s_params), np.asarray(r2.s_params)), (
        "public waveguide path is not run-to-run deterministic — the "
        "settling-witness identity check cannot be interpreted")
    assert np.array_equal(np.asarray(r1.settling_db), np.asarray(r2.settling_db))
