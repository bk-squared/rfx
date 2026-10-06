"""Declared-span validation of the masks collected by the shared builder."""
from rfx.core.jax_utils import is_tracer
from rfx.geometry.rasterize_grid import assert_declared_span_is_filled


def check_pad_fill(sim, grid, geometry_masks, *, record=None):
    """Use the same projected hi-face rule on uniform and graded grids."""
    if sim._boundary not in ("cpml", "upml") or sim._cpml_layers <= 0:
        return
    masks = dict(geometry_masks)
    for entry in sim._geometry:
        mask = masks.get(id(entry))
        if mask is not None and not is_tracer(mask):
            assert_declared_span_is_filled(
                entry.material_name, entry.shape, mask, grid,
                sim._unresolved_domain, record=record)


def report_pad_fill(sim, issues):
    """Expose the audit object's findings without assembling a second model."""
    from rfx.preflight._common import PreflightIssue
    ctx = sim._campaign_ctx()
    realized = None if ctx.error else ctx.realized()
    for row in getattr(realized, "pad_fill_findings", ()):
        issues.append(PreflightIssue(
            f"{row['entity']!r} is declared out to the {row['face']} face but "
            f"its rasterized mask stops {row['empty_interior_nodes']} interior "
            "nodes short; the absorber pad would hold vacuum (#1070).",
            severity="error", code="declared-span-short-of-padded-face"))
