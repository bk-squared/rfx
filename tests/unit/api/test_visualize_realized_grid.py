"""Grid selection and physical coordinates agree with nonuniform solves."""

from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rfx import Box, Simulation  # noqa: E402


def _graded_sim(axis):
    profile = ([0.0005, 0.0015, 0.002] if axis == "z"
               else [0.0005, 0.0015, 0.0015, 0.0005])
    sim = Simulation(
        freq_max=10e9, domain=(0.004, 0.004, 0.004), dx=0.0005,
        boundary="pec", **{f"d{axis}_profile": np.array(profile)},
    )
    sim.add_material("dielectric", eps_r=4.0)
    sim.add(Box((0, 0, 0), (0.004, 0.004, 0.004)), material="dielectric")
    return sim


@pytest.mark.parametrize("graded_axis", ["x", "z"])
def test_geometry_slice_draws_realized_cell_widths(graded_axis):
    from rfx.visualize import plot_geometry_2d_slice

    sim = _graded_sim(graded_axis)
    fig = plot_geometry_2d_slice(sim, axis=1)
    try:
        mesh = fig.axes[0].collections[0]
        coordinates = mesh.get_coordinates()
        rendered = coordinates[0, :, 0] if graded_axis == "x" else coordinates[:, 0, 1]
        expected = [0.0, 0.5, 2.0, 3.5, 4.0] if graded_axis == "x" else [0.0, 0.5, 2.0, 4.0]
        np.testing.assert_allclose(rendered, expected, atol=1e-6)
        np.testing.assert_allclose(np.asarray(mesh.get_array()), 4.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("three_dimensional", [True, False])
def test_screenshot_uses_nonuniform_physical_coordinates(monkeypatch, tmp_path, three_dimensional):
    import rfx.visualize3d as visual

    sim = _graded_sim("z")
    grid = sim._build_nonuniform_grid()
    state = SimpleNamespace(ez=np.ones(grid.shape))
    monkeypatch.setattr(visual, "HAS_MPL3D", three_dimensional)
    monkeypatch.setattr(visual, "_has_mpl", lambda: True)
    if three_dimensional:
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        original = Axes3D.plot_surface
        surfaces = []

        def capture(self, x, y, z, *args, **kwargs):
            surfaces.append(np.asarray(z))
            return original(self, x, y, z, *args, **kwargs)

        monkeypatch.setattr(Axes3D, "plot_surface", capture)
    else:
        from matplotlib.axes import Axes
        original = Axes.pcolormesh
        meshes = []

        def capture(self, *args, **kwargs):
            mesh = original(self, *args, **kwargs)
            meshes.append(mesh)
            return mesh

        monkeypatch.setattr(Axes, "pcolormesh", capture)

    output = visual.save_screenshot(sim, state, filename=str(tmp_path / "graded"), dpi=40)
    assert (tmp_path / "graded.png").stat().st_size > 100
    assert output.endswith("graded.png")
    if three_dimensional:
        np.testing.assert_allclose(surfaces[1][0], [0.0, 0.5, 2.0, 4.0], atol=1e-6)
    else:
        # Colorbar meshes are interspersed; locate the x-z field panel by
        # its title rather than depending on Matplotlib's internal calls.
        xz = next(mesh for mesh in meshes if mesh.axes.get_title().startswith("ez @ y="))
        np.testing.assert_allclose(xz.get_coordinates()[:, 0, 1],
                                   [0.0, 0.5, 2.0, 4.0], atol=1e-6)


@pytest.mark.parametrize("graded_axis", ["x", "y"])
def test_farfield_selects_realized_grid_with_only_transverse_profile(monkeypatch, graded_axis):
    pytest.importorskip("plotly")
    from rfx.nonuniform import NonUniformGrid
    from rfx.visualize import visualize_farfield_3d

    sim = _graded_sim(graded_axis)
    result = SimpleNamespace(ntff_data=object(), ntff_box=object())

    class GridCaptured(Exception):
        pass

    def capture(_data, _box, grid, *_angles):
        assert isinstance(grid, NonUniformGrid)
        assert grid.shape["xyz".index(graded_axis)] == 5
        raise GridCaptured

    monkeypatch.setattr("rfx.farfield.compute_far_field", capture)
    with pytest.raises(GridCaptured):
        visualize_farfield_3d(result, sim)
