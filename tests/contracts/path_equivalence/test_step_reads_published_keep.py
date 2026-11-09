"""Whole-grid, design-window and graded occupancy give the same probe record."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


@pytest.mark.parametrize("comparison", ["design-window", "graded"])
def test_occupancy_probe_records_match_public_paths(comparison):
    def scene(graded=False):
        sim = Simulation(freq_max=15e9, domain=(.022, .016, .012), dx=.001,
                         boundary="pec", **({"dy_profile": np.full(16, .001)} if graded else {}))
        sim.add_source((.005, .007, .005), "ez", amplitude_kind="field")
        for position in ((.013, .008, .006), (.017, .010, .007)):
            sim.add_probe(position, "ez")
        return sim

    sim = scene()
    grid = sim.freeze_mesh()
    assert grid.shape == (23, 17, 13)
    cells = np.zeros(grid.shape, np.float32)
    block = (slice(10, 12), slice(7, 9), slice(5, 7))
    cells[block] = .2
    options = dict(n_steps=60, checkpoint=False, skip_preflight=True)
    expected = np.asarray(sim.forward(pec_occupancy_override=jnp.asarray(cells), **options).time_series)
    empty = np.asarray(sim.forward(**options).time_series)
    if comparison == "design-window":
        actual = np.asarray(sim.forward(
            design_box=((.010, .007, .005), (.011, .008, .006)),
            design_occupancy_override=jnp.asarray(cells[block]), **options).time_series)
    else:
        graded = scene(graded=True)
        assert graded.freeze_mesh().shape == grid.shape
        actual = np.asarray(graded.forward(pec_occupancy_override=jnp.asarray(cells), **options).time_series)
    peak = float(np.max(np.abs(expected)))
    # The window update and the whole-grid rule are two compiled programs (bitwise at kernel level in
    # float64, tests/unit/autodiff/test_design_box_tape_nu_occupancy.py): the two-kernel bar of
    # tests/unit/nonuniform/test_nonuniform_face_depths.py:94 applies; graded stays at 9 ULP.
    bar = (60 * 10 * float(np.finfo(np.float32).eps) * peak if comparison == "design-window"
           else 9 * float(np.spacing(np.float32(peak))))
    error = float(np.max(np.abs(actual - expected)))
    control = float(np.max(np.abs(expected - empty)))
    print(f"{comparison}: mesh={grid.shape} steps=60 peak={peak:.17g} error={error:.17g} bar={bar:.17g} occupancy_free_difference={control:.17g}")
    assert np.isfinite(actual).all() and peak > 0
    assert control > bar, f"occupancy has no resolved effect: {control=} <= {bar=}"
    assert error <= bar, f"{comparison} record error {error:.17g} exceeds bar {bar:.17g}"
