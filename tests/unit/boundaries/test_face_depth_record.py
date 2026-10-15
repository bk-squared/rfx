"""The depth record validates all faces before suppressing padding."""
import numpy as np
import pytest

from rfx.boundaries.depths import Kind, resolve_face_depths
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import Grid
from rfx.nonuniform import make_nonuniform_grid


@pytest.mark.parametrize("layers,axes", [({"x_lo": 9}, "y"), ({"x_lo": 1.5}, "xyz")])
def test_builders_reject_invalid_depth_identically(layers, axes):
    messages = []
    for build in (
        lambda: Grid(1e9, (.02,) * 3, dx=.001, cpml_layers=8,
                     cpml_axes=axes, face_layers=layers),
        lambda: make_nonuniform_grid((.02, .02), np.full(20, .001), .001,
                                    cpml_layers=8, cpml_axes=axes, face_layers=layers),
    ):
        with pytest.raises(ValueError, match="must be an integer between 0 and cpml_layers=8") as error:
            build()
        messages.append(str(error.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize("mode,z_kind", [("3d", Kind.ABSORBER), ("2d_tmz", Kind.PEC), ("2d_tez", Kind.PMC)])
def test_declared_and_realized_depths(mode, z_kind):
    spec = BoundarySpec(x=Boundary("cpml", "cpml", 8, 16), y="pec",
                        z=Boundary("cpml", "cpml", 12, 12))
    records = resolve_face_depths(spec, budget=16, mode=mode)
    assert tuple(r.declared for r in records) == (8, 16, 16, 16, 12, 12)
    assert tuple(r.realized for r in records) == (8, 16, 0, 0) + ((12, 12) if mode == "3d" else (0, 0))
    assert records[-1].kind == z_kind


def test_wall_periodic_and_nonabsorbing_axes_have_zero_pads():
    records = resolve_face_depths(budget=8, absorbing_axes="xy", periodic_axes="x",
                                  pec_faces={"y_lo"}, pmc_faces={"y_hi"})
    assert tuple(r.kind for r in records) == (
        Kind.PERIODIC, Kind.PERIODIC, Kind.PEC, Kind.PMC, Kind.ABSORBER, Kind.ABSORBER)
    assert all(r.realized == 0 and r.declared == 8 for r in records)
