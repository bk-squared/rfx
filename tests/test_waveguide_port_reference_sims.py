"""Tests for the public ``port_reference_sims`` plumbing on
``compute_waveguide_s_matrix`` — per-port matched-straight-guide references
for interior-PEC multi-port junctions (T-junctions / branches / septa).

Coverage
--------
* Guard raises (milliseconds — they fire BEFORE any FDTD): normalize must be
  ``'flux'``; one reference Simulation per waveguide port; each reference grid
  must match the device grid (shape + dx); multimode and eps/sigma-override
  combinations are unsupported.
* Advisory warning: a compact T-junction whose probe planes sit on top of the
  junction must fire the clearance advisory (band kept below the TE20 cutoff so
  the advisory is not skipped).
* Behavioral witness: on the compact T-junction the matched reference lowers
  ``|S11|`` (vacuum-reference blow-up 9.8 -> 3.1 here; |S11| band-mean well
  below 1) yet the overall matrix stays NON-physical (max|S| > 1.05), locking
  the necessary-but-not-sufficient finding. Companion committed evidence lives
  at ``tests/fixtures/waveguide_tjunction_e4/`` /
  ``tests/test_waveguide_tjunction_e4e5_gates.py``.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.api import Simulation
from rfx.geometry.csg import Box


# --------------------------------------------------------------------------
# Tiny two-port builders — used only by the guard tests, which raise before
# any FDTD, so the run parameters below are never exercised.
# --------------------------------------------------------------------------

def _tiny_two_port(*, dx=0.004, n_modes=1):
    s = Simulation(
        freq_max=10e9, domain=(0.12, 0.04, 0.02),
        boundary="cpml", cpml_layers=10, dx=dx,
    )
    common = dict(
        mode=(1, 0), mode_type="TE", freqs=jnp.linspace(4.5e9, 8e9, 3),
        f0=6e9, ref_offset=3, probe_offset=8, n_modes=n_modes,
    )
    s.add_waveguide_port(0.01, direction="+x", name="a", **common)
    s.add_waveguide_port(0.11, direction="-x", name="b", **common)
    return s


@pytest.mark.parametrize("normalize", [False, True])
def test_port_reference_sims_requires_flux(normalize):
    s = _tiny_two_port()
    refs = [_tiny_two_port(), _tiny_two_port()]
    with pytest.raises(ValueError, match="requires normalize='flux'"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize=normalize, port_reference_sims=refs,
        )


def test_port_reference_sims_wrong_length_raises():
    s = _tiny_two_port()
    with pytest.raises(ValueError, match="one Simulation per waveguide port"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux", port_reference_sims=[_tiny_two_port()],
        )


def test_port_reference_sims_mismatched_grid_raises():
    s = _tiny_two_port(dx=0.004)
    bad = _tiny_two_port(dx=0.002)  # different dx -> different grid shape
    with pytest.raises(ValueError, match="must match the device grid"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux",
            port_reference_sims=[bad, _tiny_two_port()],
        )


def test_port_reference_sims_multimode_raises():
    s = _tiny_two_port(n_modes=2)
    refs = [_tiny_two_port(), _tiny_two_port()]
    with pytest.raises(NotImplementedError, match="multimode"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux", port_reference_sims=refs,
        )


def test_port_reference_sims_eps_override_combo_raises():
    s = _tiny_two_port()
    grid = s._build_grid()
    with pytest.raises(NotImplementedError, match="eps_override"):
        s.compute_waveguide_s_matrix(
            n_steps=10, normalize="flux",
            eps_override=jnp.ones(grid.shape),
            port_reference_sims=[_tiny_two_port(), _tiny_two_port()],
        )


# --------------------------------------------------------------------------
# Compact T-junction builders (geometry copied from the SKIPPED
# test_api.py::test_waveguide_branch_junction_mixed_normals_reciprocal_through_api).
# The main guide runs horizontally (y in [0.04, 0.08]); the top arm opens at
# x in [0.04, 0.08], y in [0.08, 0.12].
# --------------------------------------------------------------------------

def _tj_common(freqs, f0):
    return dict(
        mode=(1, 0), mode_type="TE", freqs=freqs, f0=f0,
        ref_offset=3, probe_offset=15, z_range=(0.00, 0.02),
    )


def _tj_sim():
    return Simulation(
        freq_max=10e9, domain=(0.12, 0.12, 0.02),
        boundary="cpml", cpml_layers=10, dx=0.002,
    )


def _tj_device(freqs, f0):
    s = _tj_sim()
    s.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    s.add(Box((0.0, 0.08, 0.0), (0.04, 0.12, 0.02)), material="pec")
    s.add(Box((0.08, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = _tj_common(freqs, f0)
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return s


def _tj_ref_horizontal(freqs, f0):
    """Straight horizontal guide (walls y in [0,0.04] and [0.08,0.12], full x)
    — the matched continuation for the left and right ports (no top arm)."""
    s = _tj_sim()
    s.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    s.add(Box((0.0, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = _tj_common(freqs, f0)
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return s


def _tj_ref_vertical(freqs, f0):
    """Straight vertical guide (walls x in [0,0.04] and [0.08,0.12], full y)
    — the matched continuation for the top port (no horizontal arm)."""
    s = _tj_sim()
    s.add(Box((0.0, 0.0, 0.0), (0.04, 0.12, 0.02)), material="pec")
    s.add(Box((0.08, 0.0, 0.0), (0.12, 0.12, 0.02)), material="pec")
    common = _tj_common(freqs, f0)
    s.add_waveguide_port(0.01, y_range=(0.04, 0.08), direction="+x", name="left", **common)
    s.add_waveguide_port(0.11, y_range=(0.04, 0.08), direction="-x", name="right", **common)
    s.add_waveguide_port(0.11, x_range=(0.04, 0.08), direction="-y", name="top", **common)
    return s


def _tj_refs(freqs, f0):
    return [
        _tj_ref_horizontal(freqs, f0),   # left  (+x)
        _tj_ref_horizontal(freqs, f0),   # right (-x)
        _tj_ref_vertical(freqs, f0),     # top   (-y)
    ]


def test_port_reference_sims_clearance_advisory_fires():
    """Probe planes sitting on top of the junction must fire the clearance
    advisory. The band is kept below the TE20 cutoff (fc2 = C0/a = 7.5 GHz for
    a = 0.04 m) so the advisory is not skipped for an in-band higher mode."""
    freqs = jnp.linspace(4.5e9, 6.5e9, 3)
    f0 = 5.5e9
    with pytest.warns(UserWarning) as record:
        _tj_device(freqs, f0).compute_waveguide_s_matrix(
            num_periods=8, normalize="flux", port_reference_sims=_tj_refs(freqs, f0),
        )
    messages = [str(w.message) for w in record]
    assert any("clearance" in m for m in messages), messages


def test_port_reference_sims_compact_junction_necessary_not_sufficient():
    """A/B witness on the compact T-junction: the matched reference lowers
    |S11| (vacuum blow-up -> physical diagonal) but the overall matrix stays
    non-physical (max|S| > 1.05) and the passivity self-check still fires —
    locking the necessary-but-not-sufficient finding in BOTH directions.

    Full band (includes the TE20-propagating region) so the vacuum reference
    blows up the way the 2026-07-06 verification documented (max|S| ~ 9.8)."""
    freqs = jnp.linspace(4.5e9, 8.0e9, 6)
    f0 = 6e9
    num_periods = 30

    s_vac = np.asarray(
        _tj_device(freqs, f0)
        .compute_waveguide_s_matrix(num_periods=num_periods, normalize="flux")
        .s_params
    )
    # The extractor's passivity self-check MUST still fire with references —
    # the compact-geometry S-matrix is still non-physical (locked).
    with pytest.warns(UserWarning, match="passiv"):
        result = _tj_device(freqs, f0).compute_waveguide_s_matrix(
            num_periods=num_periods, normalize="flux",
            port_reference_sims=_tj_refs(freqs, f0),
        )
    s_ref = np.asarray(result.s_params)

    s11_vac = float(np.mean(np.abs(s_vac[0, 0, :])))
    s11_ref = float(np.mean(np.abs(s_ref[0, 0, :])))
    max_vac = float(np.max(np.abs(s_vac)))
    max_ref = float(np.max(np.abs(s_ref)))

    # Direction 1 — the matched reference fixes the reflection diagonal.
    assert s11_ref < s11_vac            # lower than the vacuum reference
    assert s11_ref < 1.0                # physical reflection band-mean
    # Direction 2 — the overall matrix is still non-physical (compact geometry).
    assert max_ref > 1.05               # non-passive residual remains
    assert max_ref < max_vac            # but the blow-up is reduced
