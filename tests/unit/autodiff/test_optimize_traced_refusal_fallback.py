"""optimize(jit=True) falls back on rfx's traced-structure refusal type itself."""
from rfx._traced_refusal import TracedStructureRefusal
def test_every_named_traced_refusal_is_the_rfx_type():
    from rfx.geometry.rasterize_grid import TracedShapeDimensionError
    from rfx.model.graded_ports import TracedWireLiveCellsError
    assert issubclass(TracedShapeDimensionError, TracedStructureRefusal)
    assert issubclass(TracedWireLiveCellsError, TracedStructureRefusal)
