"""Published factors against an independent float64 four-cell product."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.model.occupancy import build_edge_keep


@pytest.mark.parametrize("case", ["block", "random", "random_periodic"])
def test_published_factor_within_eight_reference_ulps(case, record_property):
    shape = (23, 17, 13)
    periodic = (False, case == "random_periodic", False)
    cells = np.zeros(shape, np.float32)
    if case == "block":
        cells[10:12, 7:9, 5:7] = .2
    else:
        cells = np.random.default_rng(202).uniform(0, .85, shape).astype(np.float32)
    cells64 = cells.astype(np.float64)
    references = []
    for component in range(3):
        transverse = [axis for axis in range(3) if axis != component]
        product = np.ones(shape, np.float64)
        for first, second in ((0, 0), (1, 0), (0, 1), (1, 1)):
            shifted = cells64.copy()
            for axis, offset in zip(transverse, (first, second)):
                if offset:
                    shifted = np.roll(shifted, 1, axis=axis)
                    if not periodic[axis]:
                        face = [slice(None)] * 3
                        face[axis] = 0
                        shifted[tuple(face)] = 0.
            product *= 1. - shifted
        references.append(product.astype(np.float32))

    actual = build_edge_keep(jnp.asarray(cells), periodic=periodic)
    largest = 0.
    for got, reference in zip(actual, references):
        got = np.asarray(got)
        error = np.abs(got.astype(np.float64) - reference.astype(np.float64))
        spacing = np.spacing(reference)
        largest = max(largest, float(np.max(error / spacing)))
        assert np.all(error <= 8 * spacing)
        assert np.all(got[reference == 1] == 1)
    record_property("largest_reference_ulps", largest)

    masks = tuple(np.zeros(shape, bool) for _ in range(3))
    for component, mask in enumerate(masks):
        mask[10 + component, 6:10, 4:8] = True
    masked = build_edge_keep(jnp.asarray(cells), periodic=periodic,
                             sheet_edge_masks=tuple(jnp.asarray(m) for m in masks))
    for got, reference, mask in zip(masked, references, masks):
        got = np.asarray(got)
        assert np.all(got[mask] == 0)
        assert np.all(got[(reference == 1) & ~mask] == 1)
        error = np.abs(got[~mask].astype(np.float64) - reference[~mask].astype(np.float64))
        assert np.all(error <= 8 * np.spacing(reference[~mask]))
