"""A periodic MSL feed sees the conductor immediately across its seam."""
import sys

import numpy as np
import pytest

from rfx import Box, Simulation


def _model(graded, shift):
    profile = {"dz_profile": np.array([.001] * 4 + [.0005] * 8 + [.0015] * 12)} if graded else {}
    sim = Simulation(freq_max=5e9, domain=(.024, .024, .026), dx=.001,
                     cpml_layers=0, boundary={"x": "periodic", "y": "cpml", "z": "cpml"},
                     **profile)
    sim.add(Box((0, 0, .004), (.024, .024, .004)), material="pec")
    sim.add_material("substrate", eps_r=3)
    sim.add(Box((0, 0, .004), (.024, .024, .008)), material="substrate")
    # The full-period ground and substrate are invariant under translation.
    sim.add(Box(((16 - .35 + shift) * .001, (10 - .35) * .001, .008),
                ((24 + shift) * .001, (14 + .35) * .001, .008)), material="pec")
    sim.add_msl_port(position=(shift * .001, .012, .004),
                     width=.004, height=.004, direction="-x")
    # Off the trace's y=12 mm symmetry plane and all domain midplanes;
    # x=23 mm translates to x=4 mm, with z=6 mm inside the substrate.
    sim.add_probe((((23 + shift) % 24) * .001, .011, .006), "ez")
    return sim


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_msl_seam_probe_record_is_translation_invariant(graded):
    records = []
    for shift in (0, 5):
        result = _model(graded, shift).run(n_steps=32, skip_preflight=True)
        trace = np.asarray(result.time_series)
        shape = tuple(result.grid.shape)
        # Quote the realized lattice: x has 24 unique periodic nodes.
        assert shape == ((24, 25, 25) if graded else (24, 25, 27)), shape
        assert trace.shape == (32, 1)
        assert trace.dtype == np.float32
        assert np.all(np.isfinite(trace))
        peak = float(np.max(np.abs(trace)))
        assert peak > 0, f"graded={graded}, shift={shift}: empty probe signal"
        records.append(trace)

    seam, shifted = records
    difference = float(np.max(np.abs(seam - shifted)))
    mismatches = int(np.count_nonzero(seam.view(np.uint32) != shifted.view(np.uint32)))
    print(f"MSL mesh={'graded' if graded else 'uniform'} shape={shape} steps=32 "
          f"quantity=probe_translation max_abs_difference={difference:.17g} "
          f"bit_mismatches={mismatches} bar=0 seam_peak={np.max(np.abs(seam)):.17g} "
          f"shifted_peak={np.max(np.abs(shifted)):.17g}", file=sys.stderr)
    # A cell-aligned translation on the same kernel has the spec's bitwise bar;
    # this does not compare the uniform and graded kernels with each other.
    np.testing.assert_array_equal(
        seam.view(np.uint32), shifted.view(np.uint32),
        err_msg=f"MSL seam/+5-cell probe record differs: graded={graded}, {difference=}, {mismatches=}")
