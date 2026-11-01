"""Prepared-product entries for the existing array-based time steppers.

These entries unpack the solve-owned operands. Metadata holds a weak reference
back to this product; no reporting assembly participates in stepping.
"""
from dataclasses import fields


def stage(product, *, grid=None, materials=None, **updates):
    """Complete a prepared product with the runner's actual stage operands."""
    owner = product._prepared_owner or product
    if owner is not product:
        for item in fields(product):
            if item.name != "_prepared_owner":
                object.__setattr__(owner, item.name, getattr(product, item.name))
    if grid is not None:
        updates["grid"] = grid
    if materials is not None:
        updates["materials"] = materials
    for name, value in updates.items():
        object.__setattr__(owner, name, value)
    return owner


def release_dense(product):
    """Keep the prepared identity while transferring dense arrays to slabs."""
    for name, value in dict(materials=None, pec_cells=None, pec_edges=None,
        sheets=(), wires=(), sheet_impedance=(), geometry_masks={}, assembly_entries=(),
        assembly=(), sheet_operator=None, _prepared_owner=None).items():
        object.__setattr__(product, name, value)


def advance(product, stepper, *args, _operand_layout="grid", **kwargs):
    """Step with the product taken by the runner, using its final operands."""
    args = list(args)
    layout = _operand_layout
    if layout == "grid":
        args[:2] = product.grid, product.materials
        if "pec_edge_masks" in kwargs:
            kwargs["pec_edge_masks"] = product.pec_edges
    elif layout == "adi3d":
        args[6:8] = product.materials.eps_r, product.materials.sigma
        kwargs["pec_edge_masks"] = product.pec_edges
    elif layout == "adi2d":
        args[3:5] = product.materials.eps_r[:, :, 0], product.materials.sigma[:, :, 0]
    elif layout == "slab":
        kwargs["sharded_materials"] = product.materials
        kwargs["sharded_pec_mask"] = product.pec_edges
    elif layout == "graded_slab":
        args[:3] = product.grid, product.materials, product.pec_edges
    elif layout == "disjoint":
        args[1] = product.kernel_context[0]
    return stepper(*args, **kwargs)
