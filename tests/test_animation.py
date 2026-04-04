"""Field animation export tests.

Validates GIF creation from both 3D and 2D snapshot data,
custom parameters, error handling, and dict-based input.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from rfx import Simulation, SnapshotSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_result_3d():
    """Run a small 3D simulation with full-volume snapshots."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.02),
        boundary="pec",
    )
    sim.add_probe((0.01, 0.01, 0.01), "ez")
    snap = SnapshotSpec(components=("ez",))
    result = sim.run(n_steps=20, snapshot=snap)
    return result


@pytest.fixture
def sim_result_2d():
    """Run a small simulation with 2D-sliced snapshots."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.02),
        boundary="pec",
    )
    sim.add_probe((0.01, 0.01, 0.01), "ez")
    snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=5)
    result = sim.run(n_steps=20, snapshot=snap)
    return result


@pytest.fixture
def fake_snapshots_3d():
    """Synthetic 3D snapshot dict (no simulation needed)."""
    rng = np.random.default_rng(42)
    return {
        "ez": rng.standard_normal((10, 8, 8, 8)).astype(np.float32),
    }


@pytest.fixture
def fake_snapshots_2d():
    """Synthetic 2D snapshot dict (no simulation needed)."""
    rng = np.random.default_rng(42)
    return {
        "ez": rng.standard_normal((10, 8, 8)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# GIF export from real simulations
# ---------------------------------------------------------------------------

class TestGifExport:

    def test_gif_from_3d_snapshots(self, sim_result_3d, tmp_path):
        """GIF export from full-volume snapshots should produce a valid file."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "field_3d.gif")
        result_path = save_field_animation(sim_result_3d, out)
        assert os.path.exists(result_path)
        assert result_path.endswith(".gif")
        assert os.path.getsize(result_path) > 100

    def test_gif_from_2d_snapshots(self, sim_result_2d, tmp_path):
        """GIF export from 2D-sliced snapshots should work."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "field_2d.gif")
        result_path = save_field_animation(sim_result_2d, out)
        assert os.path.exists(result_path)
        assert result_path.endswith(".gif")
        assert os.path.getsize(result_path) > 100

    def test_gif_auto_extension(self, sim_result_2d, tmp_path):
        """If no extension given, .gif is appended."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "no_ext")
        result_path = save_field_animation(sim_result_2d, out)
        assert result_path.endswith(".gif")
        assert os.path.exists(result_path)


# ---------------------------------------------------------------------------
# Custom parameters
# ---------------------------------------------------------------------------

class TestCustomParams:

    def test_custom_colormap(self, fake_snapshots_2d, tmp_path):
        """Custom colormap should not raise."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "viridis.gif")
        result_path = save_field_animation(
            fake_snapshots_2d, out, colormap="viridis", fps=10,
        )
        assert os.path.exists(result_path)

    def test_custom_vmin_vmax(self, fake_snapshots_2d, tmp_path):
        """Explicit vmin/vmax should be respected."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "custom_range.gif")
        result_path = save_field_animation(
            fake_snapshots_2d, out, vmin=-0.5, vmax=0.5,
        )
        assert os.path.exists(result_path)

    def test_custom_figsize_dpi(self, fake_snapshots_2d, tmp_path):
        """Custom figsize and dpi should produce a valid file."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "custom_size.gif")
        result_path = save_field_animation(
            fake_snapshots_2d, out, figsize=(8, 6), dpi=72,
        )
        assert os.path.exists(result_path)

    def test_interval_stride(self, fake_snapshots_2d, tmp_path):
        """Frame stride should reduce the number of frames."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "strided.gif")
        result_path = save_field_animation(
            fake_snapshots_2d, out, interval=3,
        )
        assert os.path.exists(result_path)


# ---------------------------------------------------------------------------
# Slice axis selection (3D data)
# ---------------------------------------------------------------------------

class TestSliceAxis:

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_all_slice_axes(self, fake_snapshots_3d, tmp_path, axis):
        """All three slice axes should produce valid GIFs."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / f"slice_{axis}.gif")
        result_path = save_field_animation(
            fake_snapshots_3d, out, slice_axis=axis,
        )
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 100

    def test_explicit_slice_index(self, fake_snapshots_3d, tmp_path):
        """Explicit slice_index should work for 3D data."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "explicit_idx.gif")
        result_path = save_field_animation(
            fake_snapshots_3d, out, slice_axis="x", slice_index=2,
        )
        assert os.path.exists(result_path)


# ---------------------------------------------------------------------------
# Dict-based input
# ---------------------------------------------------------------------------

class TestDictInput:

    def test_dict_input(self, fake_snapshots_2d, tmp_path):
        """Passing a raw dict should work (no Result wrapper needed)."""
        from rfx.animation import save_field_animation

        out = str(tmp_path / "from_dict.gif")
        result_path = save_field_animation(fake_snapshots_2d, out)
        assert os.path.exists(result_path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    def test_missing_snapshots(self, tmp_path):
        """Should raise ValueError when result has no snapshots."""
        from rfx.animation import save_field_animation

        class FakeResult:
            snapshots = None

        with pytest.raises(ValueError, match="No snapshots found"):
            save_field_animation(FakeResult(), str(tmp_path / "fail.gif"))

    def test_missing_component(self, fake_snapshots_2d, tmp_path):
        """Should raise ValueError for absent component."""
        from rfx.animation import save_field_animation

        with pytest.raises(ValueError, match="not in snapshots"):
            save_field_animation(
                fake_snapshots_2d, str(tmp_path / "fail.gif"),
                component="hx",
            )

    def test_invalid_slice_axis(self, fake_snapshots_3d, tmp_path):
        """Should raise ValueError for bad slice_axis."""
        from rfx.animation import save_field_animation

        with pytest.raises(ValueError, match="slice_axis"):
            save_field_animation(
                fake_snapshots_3d, str(tmp_path / "fail.gif"),
                slice_axis="w",
            )

    def test_empty_dict(self, tmp_path):
        """Should raise ValueError for empty dict."""
        from rfx.animation import save_field_animation

        with pytest.raises(ValueError, match="No snapshots found"):
            save_field_animation({}, str(tmp_path / "fail.gif"))

    def test_bad_result_type(self, tmp_path):
        """Should raise ValueError for non-result, non-dict input."""
        from rfx.animation import save_field_animation

        with pytest.raises(ValueError, match="result must be"):
            save_field_animation("not_a_result", str(tmp_path / "fail.gif"))
