from pathlib import Path
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/showcase"))
import design_lens as lens


def test_lens_graded_mesh():
    """Pins the page's mesh: cubic cells at the plate's four edges, and with `--boundary nearest`
    a cell that straddles two lens pixels takes the permittivity of the pixel holding its centre."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = lens.Model(1e-3, boundary="arith")
    nodes = model.nodes
    plane = np.argmin(abs(nodes[2] - .018))
    normal = np.diff(nodes[2][plane - 1:plane + 2])
    for line in nodes[:2]:
        for edge, lo, hi in ((.048, .04735, .04835), (.078, .07765, .07865)):
            k = np.searchsorted(line, edge, side="right") - 1
            np.testing.assert_allclose(line[k:k + 2], [lo, hi], atol=1e-12, rtol=0)
            width = line[k + 1] - line[k]
            np.testing.assert_allclose(width / .001, 1., atol=1e-9, rtol=0)
            np.testing.assert_allclose(normal / width, 1., atol=1e-9, rtol=0)
    expected_faces = (np.r_[0., 3., 123., 126., 48., 78., 18. + 3. * np.arange(31)],
                      np.r_[0., 3., 123., 126., 48., 78., 18. + 3. * np.arange(31)],
                      np.r_[0., 3., 18., 117., 120., 72. + 3. * np.arange(11)])
    for axis, faces in enumerate(expected_faces):
        np.testing.assert_allclose(lens.FACES[axis], faces * .001, atol=1e-15, rtol=0)
        assert nodes[axis].dtype == np.float64
        for face in faces:
            on_node = np.min(abs(nodes[axis] - face * .001)) <= 1e-12
            assert on_node == (axis == 2 or face not in (48., 78.))
    assert tuple(b - a for a, b in zip(model.lo, model.hi)) == (92, 92, 30)
    free = np.broadcast_to(np.arange(15)[:, None, None] / 20 +
                           np.arange(15)[None, :, None] / 40 + 1.1, (15, 15, 10)).copy()
    crossing = []
    for axis in range(2):
        line = nodes[axis][model.lo[axis]:model.hi[axis] + 1]
        crossing.append(np.any((line[:-1, None] < np.array([.048, .078])) &
                               (line[1:, None] > np.array([.048, .078])), axis=1))
    mask = np.broadcast_to(crossing[0][:, None, None] | crossing[1][None, :, None], (92, 92, 30))
    arrays = []
    for boundary in ("nearest", "arith", "harm"):
        model.boundary = boundary
        model.weights = tuple(lens.pixel_weights(line[a:b + 1], faces * .001, boundary)
                              for line, a, b, faces in zip(nodes, model.lo, model.hi,
                                  (18. + 3. * np.arange(31), 18. + 3. * np.arange(31),
                                   72. + 3. * np.arange(11))))
        arrays.append(model.cells(free))
    for i, left in enumerate(arrays):
        for right in arrays[i + 1:]:
            np.testing.assert_allclose(left[~mask], right[~mask], rtol=1e-6, atol=0)
            assert np.any(abs(left[mask] - right[mask]) > 1e-6)
    ix = np.argmin(abs(nodes[0] - .04735)) - model.lo[0]
    iy = np.argmin(abs(nodes[1] - .064)) - model.lo[1]
    iz = np.argmin(abs(nodes[2] - .073)) - model.lo[2]
    left, right = 1.1 + 5 / 20, 1.1 + 4 / 20
    expected = (left, .65 * left + .35 * right, 1 / (.65 / left + .35 / right))
    np.testing.assert_allclose([a[ix, iy, iz] for a in arrays], expected, rtol=1e-12, atol=1e-12)
