"""PR A float64 model invariant, rfx#2613; DESIGN A1-A5 and A2b-A2g.

R1 reads realized_operands at boundaries.tfsf.admit_setup; R2 its pre-widening
mesh; R3 init_cpml's return; R4 recorded_scan inputs and closed-body constants;
R5 pole_component_weights returns, labeled by their consuming _realize_poles.
Every scene declares 200 steps (A2f); capture stops before the first step.
P1 applies everywhere, P2 only at R4/R5. Equation outputs/literals are not read
(A2b). P2 is computed from bits independently in float64_readpoints.py.
Limit: P2 tests the whole array; a mostly-float32 table with one true float64
element does not flag, because not every element round-trips through float32.

FLOAT32_UNSEEN_2613 records the unreached FIX sites and the reached conformal
face areas consumed before the declared reads. U-smooth's True declaration
selects full node tensors, already float64, not the old Kottke cast paths.
UPML source coefficients (materials.py:692,694) fall with A1's refusal.

Each pending name pins its per-scene violation count (omitted scenes mean zero).
More arrays fail as "new violation under a listed name"; fewer fail as
"fixed: lower the count or remove the entry", including fallback-name collisions.
The ceiling counts pending entries plus unseen names, not arrays, and only falls;
scripts/ci/check_file_size_ratchet.py compares its integer against the base:
never increase the integer to admit a new violation. Remove fixed observed
names; transfer an unseen name to pending only with evidence, keeping the sum
at or below the ceiling. All named exceptions must retain their census reason.
"""
from collections import Counter

import jax
import pytest
import numpy as np

from tests._x64_compat import enable_x64
from tests.contracts.float64_pending import (
    FLOAT32_BY_CONTRACT, FLOAT32_PENDING_2613, FLOAT32_UNSEEN_2613, PENDING_CEILING,
)
from tests.contracts.float64_readpoints import SCENES, capture


@pytest.fixture(scope="module")
def captured_scenes():
    """One capture per scene, shared by dtype and independent value rows."""
    cache = {}
    print(f"B_MAT_DEVICES count={len(jax.devices())} devices={jax.devices()}")

    def get(label):
        if label not in cache:
            values = {}
            with enable_x64():
                try:
                    rows = capture(label, values=values)
                except ValueError as error:
                    if not label.endswith("tfsf"):
                        raise
                    cache[label] = error
                else:
                    cache[label] = (rows, values)
            jax.clear_caches()
        if isinstance(cache[label], Exception):
            raise cache[label]
        return cache[label]

    yield get
    cache.clear()
    jax.clear_caches()


def test_float64_model_invariant(captured_scenes):
    allowed = FLOAT32_BY_CONTRACT | FLOAT32_PENDING_2613
    assert not FLOAT32_BY_CONTRACT.keys() & FLOAT32_PENDING_2613.keys()
    assert all(allowed.values()) and all(FLOAT32_UNSEEN_2613.values())
    assert all(reason and counts and set(counts) <= set(SCENES)
               and all(type(count) is int and count > 0 for count in counts.values())
               for reason, counts in FLOAT32_PENDING_2613.values())
    assert len(FLOAT32_PENDING_2613) + len(FLOAT32_UNSEEN_2613) <= PENDING_CEILING
    seen, unexpected, more, fewer = set(), [], [], []
    with enable_x64():
        for label in SCENES:
            counts = Counter()
            if label.endswith('tfsf'):
                # A2d: split once; this scene already contains only one object.
                for _ in range(2):
                    with pytest.raises(ValueError, match='update.materials.eps_r prevents an invariant periodic replacement'):
                        captured_scenes(label)
                rows = []
            else:
                rows, _ = captured_scenes(label)
            for point, name, dtype, shape, p1, p2 in rows:
                if p1 or p2:
                    seen.add(name)
                    counts[name] += 1
                    if name not in allowed:
                        unexpected.append((label, point, name, dtype, shape, 'P1' if p1 else 'P2'))
            for name, (_, expected) in FLOAT32_PENDING_2613.items():
                actual, wanted = counts[name], expected.get(label, 0)
                if actual > wanted:
                    more.append((label, name, wanted, actual))
                elif actual < wanted:
                    fewer.append((label, name, wanted, actual))
            jax.clear_caches()
    assert not unexpected, f'new float64 violations: {unexpected}'
    assert not more, f'new violation under a listed name (scene, name, expected, actual): {more}'
    assert not fewer, f'fixed: lower the count or remove the entry (scene, name, expected, actual): {fewer}'
    assert not FLOAT32_BY_CONTRACT.keys() - seen, f'fixed: remove it: {sorted(FLOAT32_BY_CONTRACT.keys() - seen)}'


@pytest.mark.parametrize('label', ['U-mat', 'G-x', 'G-yz'])
@pytest.mark.parametrize('quantity', ['eps', 'sigma'])
def test_material_edge_values(captured_scenes, label, quantity):
    from tests.contracts.float64_value_reference import assert_spacings, edge_reference
    _, data = captured_scenes(label)
    materials = data['materials']
    cells = materials.eps_r if quantity == 'eps' else materials.sigma
    for axis, actual in enumerate(getattr(materials.components, quantity)):
        assert_spacings(actual, edge_reference(cells, data['grid'], axis), 19)


@pytest.mark.parametrize('label', ['U-mat', 'G-x', 'G-yz'])
def test_pole_edge_values(captured_scenes, label):
    from tests.contracts.float64_value_reference import assert_spacings, edge_reference
    _, data = captured_scenes(label)
    assert data['poles']
    for mask, _, weights in data['poles']:
        for axis, actual in enumerate(weights):
            assert_spacings(actual, edge_reference(mask, data['grid'], axis), 19)


def test_uniform_mu_values(captured_scenes):
    from tests.contracts.float64_value_reference import assert_spacings
    _, data = captured_scenes('U-mat')
    mu = np.asarray(data['materials'].mu_r, dtype=np.float64)
    for axis, actual in enumerate(data['materials'].components.mu):
        lo = np.take(mu, np.maximum(np.arange(mu.shape[axis])-1, 0), axis=axis)
        expected = 2*lo*mu/(lo+mu)
        assert_spacings(actual, expected, 8)
        np.testing.assert_array_equal(np.asarray(actual)[lo == mu], mu[lo == mu])


def test_chi3_values(captured_scenes):
    _, data = captured_scenes('U-kerr')
    actual = np.asarray(data['updates']['kerr_chi3'])
    assert actual.dtype == np.dtype('float64')
    assert np.any(actual != 0)
    np.testing.assert_array_equal(actual[actual != 0], np.full(np.count_nonzero(actual), 2.2e-19))


def test_conformal_values(captured_scenes):
    """SC: independent polygons, zero/near-zero midpoint rule and saddle ties.

    The complete corner rule is declared in float64_value_reference's module
    docstring. Inputs are the table's sampled distances, not its fractions.
    """
    from tests.contracts.float64_value_reference import (
        assert_spacings, conformal_reference, clamp_reference,
    )
    _, data = captured_scenes('U-conformal')
    assert data['conformal']
    binding_count = 0
    for samples, table, (actual_lengths, actual_areas) in data['conformal']:
        lengths, areas = conformal_reference(samples)
        for actual, expected in zip(table.lengths, lengths):
            assert_spacings(actual, expected, 8)
        for actual, expected in zip(table.areas, areas):
            assert_spacings(actual, expected, 32)
        lengths = tuple(a[:-1, :-1, :-1] for a in lengths)
        areas = tuple(a[:-1, :-1, :-1] for a in areas)
        clamped = clamp_reference(lengths, areas)
        for raw, expected, actual, area, actual_area in zip(lengths, clamped, actual_lengths, areas, actual_areas):
            physical = (slice(1, None),)*3
            bound = (expected < raw)[physical]
            binding_count += np.count_nonzero(bound)
            assert_spacings(actual_area, area[physical], 32)
            assert_spacings(np.asarray(actual)[~bound], expected[physical][~bound], 8)
            assert_spacings(np.asarray(actual)[bound], expected[physical][bound], 64)
    assert binding_count > 0


@pytest.mark.parametrize('source', [0, 1], ids=['vacuum', 'dielectric'])
def test_uniform_source_coefficient_values(captured_scenes, source):
    from tests.contracts.float64_value_reference import assert_spacings
    _, data = captured_scenes('U-src')
    assert len(data['sources']) == 2
    _, component, dt, actual = data['sources'][source]
    assert component.lower() == 'ez'
    eps, sigma = ((1., 0.), (4., .05))[source]
    # Same declared epsilon_0 as the solver, written without calling helpers.
    eps0 = np.float64(8.8541878128e-12)
    dt = np.float64(dt)
    expected = (dt/eps0)/(np.float64(eps)+np.float64(sigma)*(dt/(2*eps0)))
    assert_spacings(actual, expected, 12)
