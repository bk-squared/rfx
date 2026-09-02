"""Ring-down settling witness on compute_msl_s_matrix (CLAUDE.md rule, made mechanical).

Root cause this guards (measured, PR #462 Sheen-1990 study, dx=200 µm):
a fixed num_periods=20 record ended while the LPF stopband was still ringing,
and the truncated single-bin DFTs produced |S| column-power poles up to ~1.8e3
— 58/120 non-passive bins — that shrank monotonically with record length
(20→60 periods: worst pole 62→8.8) while absorber depth (8→24 CPML layers)
did not move them. Nothing in the MSL path measured settling, so the result
looked like ordinary (bad) S-parameters instead of a truncation artifact.

Two tiny thru-line FDTD runs (~seconds each): a deliberately truncated record
must fail the witness loudly, a settled record must pass silently.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation


def _thru(domain_y=0.008, y_c=0.004):
    sim = Simulation(freq_max=20e9, domain=(0.012, domain_y, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, domain_y, 0.0008)), material="sub")
    sim.add(Box((0.0, y_c - 0.0006, 0.0008),
                (0.012, y_c + 0.0006, 0.0010)), material="pec")
    sim.add_msl_port(position=(0.002, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


FREQS = jnp.linspace(2e9, 18e9, 12)


def test_truncated_record_fails_the_witness_loudly():
    sim = _thru()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)

    assert res.settling_db is not None and res.settling_db.shape == (2,)
    assert np.all(np.isfinite(res.settling_db))
    # A 2-period record ends essentially at the transit peak.
    assert np.all(res.settling_db > -40.0)

    witness = [w for w in caught if "settling witness" in str(w.message)]
    assert witness, "a truncated record must warn, not just return numbers"
    msg = str(witness[0].message)
    # The warning must carry the measured value and the actionable knob.
    assert "num_periods" in msg and "-40" in msg
    assert "settling_db" in msg


def test_settled_record_passes_the_witness_silently():
    sim = _thru()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=FREQS, num_periods=40.0)

    assert res.settling_db is not None
    # A matched thru rings down fast; the witness must be deeply settled,
    # not merely under the line (guards against a witness that measures
    # the wrong window and hovers near the threshold).
    assert np.all(res.settling_db < -60.0), res.settling_db

    assert not [w for w in caught if "settling witness" in str(w.message)]


def test_witness_survives_pre_existing_user_probes():
    """The witness columns are indexed from len(self._probes) at call time; a
    wrong base would silently measure a USER probe and produce a plausible
    settling number — the worst failure class for a witness. Registering a
    user probe first exercises exactly that offset."""
    sim = _thru()
    sim.add_probe(position=(0.006, 0.004, 0.0016), component="ez")
    n_before = len(sim._probes)

    res = sim.compute_msl_s_matrix(freqs=FREQS, num_periods=40.0)

    assert len(sim._probes) == n_before, "user probes must survive untouched"
    # A settled thru must still read deeply settled through the offset base.
    assert res.settling_db is not None
    assert np.all(res.settling_db < -60.0), res.settling_db


def test_witness_probes_do_not_leak_into_the_simulation():
    sim = _thru()
    n_probes_before = len(sim._probes)
    sim.compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    assert len(sim._probes) == n_probes_before


def test_result_field_is_optional_for_backward_compatibility():
    from rfx import MSLSMatrixResult

    legacy = MSLSMatrixResult(
        S=np.zeros((2, 2, 3), dtype=complex),
        freqs=np.array([1e9, 2e9, 3e9]),
        Z0=np.zeros((2, 3), dtype=complex),
        beta=np.zeros(3, dtype=complex),
    )
    assert legacy.settling_db is None
