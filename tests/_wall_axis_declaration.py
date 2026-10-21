"""Explicit warning witnesses for fixtures whose off-node drawing is intentional."""
import math
import warnings


def assert_declared_wall_warning(sim, axes=None):
    """Check an intentional off-node fixture's warning with independent arithmetic.

    Callers may name their closed axes; otherwise use the explicit boundary. Profiled axes keep their declared profile;
    the expected warning uses ordinary half-up arithmetic, not a grid helper.
    """
    assert sim._snap == 'declared'
    # A periodic axis that is not yet whole cells is fitted by its caller after construction (a Floquet port rounds
    # the cell); an early build would meet the periodic-divisor refusal instead of the wall warning, so skip those.
    if sim._dx is not None and any(
            getattr(sim._boundary_spec, a).lo == 'periodic'
            and abs(sim._domain['xyz'.index(a)] / sim._dx - round(sim._domain['xyz'.index(a)] / sim._dx)) > 1e-6
            for a in 'xyz'):
        return sim
    if axes is None:
        axes = [a for a in 'xyz' if all(getattr(getattr(sim._boundary_spec, a), side)
                in ('pec', 'pmc') for side in ('lo', 'hi'))]
    dx = sim._dx
    assert dx is not None, 'automatic spacing must be fitted, not accepted by snap'
    expected = any(
        abs(max(1, math.floor(sim._domain['xyz'.index(a)] / dx + .5)) * dx
            / sim._domain['xyz'.index(a)] - 1) > .01
        for a in axes if getattr(sim, f'_d{a}_profile') is None
        and not (a == 'z' and sim._mode.startswith('2d')))
    # On-node variants need no extra build; preserve their existing call sequence.
    if not expected:
        return sim
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim._build_realized_grid()
    wall = [w for w in caught if 'wall axis ' in str(w.message)]
    assert bool(wall) == expected, [str(w.message) for w in caught]
    if wall:
        assert all(issubclass(w.category, UserWarning) for w in wall)
        assert all('nearest dx=' in str(w.message) for w in wall)
    return sim
