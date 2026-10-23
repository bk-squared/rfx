"""Realized spatial operands shared by model admission rules.

Enumeration and selection preserve traced values; each caller owns its
per-kind policy and any comparison against the recorded vacuum value.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import enum
from fnmatch import fnmatchcase

import numpy as np


class OperandKind(enum.Enum):
    EPS = enum.auto()
    SIGMA = enum.auto()
    MU = enum.auto()
    DEBYE = enum.auto()
    LORENTZ = enum.auto()
    DRUDE = enum.auto()
    CHI3 = enum.auto()
    PEC_VOLUME = enum.auto()
    PEC_SHEET = enum.auto()
    PEC_WIRE = enum.auto()
    IMPEDANCE_SHEET = enum.auto()
    THIN_FILM = enum.auto()
    RLC = enum.auto()
    PORT_STAMP = enum.auto()
    SOURCE_STAMP = enum.auto()


# (vacuum value / adapter, user-facing label, kinds). None marks
# configuration or observation; the leaves adapter is a container only.
OPERANDS = {
    **{name: (None, "configuration/source/observation",
              frozenset({OperandKind.SOURCE_STAMP}) if name in ("sources", "mag_sources")
              else frozenset()) for name in (
        "grid", "n_steps", "boundary", "cpml_axes", "pec_axes", "periodic",
        "tfsf", "_feature_tfsf", "sources", "mag_sources", "probes", "dft_planes",
        "flux_monitors", "waveguide_ports", "ntff", "ntff_box", "ntff_data",
        "current_moments", "field_dtype", "stencil_order", "aniso_inv_eps_smooth",
        "wire_port_sparams", "lumped_port_sparams", "wire_refplane_sparams",
        "wire_ports", "s_param_freqs", "pec_faces", "pmc_faces", "emit_time_series",
        "rlc_states")},
    "materials": ("leaves", "material", frozenset()),
    "debye": ("poles", "Debye pole medium", frozenset({OperandKind.DEBYE})),
    "lorentz": ("poles", "Drude/Lorentz pole medium", frozenset({OperandKind.LORENTZ, OperandKind.DRUDE})),
    "kerr_chi3": (0, "Kerr (chi3) medium", frozenset({OperandKind.CHI3})),
    "pec_mask": (0, "PEC volume", frozenset({OperandKind.PEC_VOLUME})),
    "pec_edge_masks": (0, "PEC edge (volume, sheet, wire or pinned sheet)",
                       frozenset({OperandKind.PEC_VOLUME, OperandKind.PEC_SHEET, OperandKind.PEC_WIRE})),
    "pec_sheets": ("realized", "PEC edges are checked through pec_edge_masks", frozenset({OperandKind.PEC_SHEET})),
    "pec_wires": ("realized", "PEC edges are checked through pec_edge_masks", frozenset({OperandKind.PEC_WIRE})),
    "pec_occupancy": (0, "PEC occupancy", frozenset({OperandKind.PEC_VOLUME})),
    "aniso_eps": (1, "anisotropic permittivity", frozenset({OperandKind.EPS})),
    "aniso_inv_eps": (1, "inverse permittivity", frozenset({OperandKind.EPS})),
    "conformal_weights": (1, "conformal PEC", frozenset({OperandKind.PEC_VOLUME})),
    "sheet_impedance": ("sheet", "surface-impedance sheet", frozenset({OperandKind.IMPEDANCE_SHEET})),
    "design_box": ("window", "design window", frozenset({OperandKind.EPS})),
    "design_occupancy": ("window", "design occupancy window", frozenset({OperandKind.EPS})),
    "lumped_rlc": ("cells", "lumped RLC element", frozenset({OperandKind.RLC})),
    "rlc_metas": ("cells", "lumped RLC element", frozenset({OperandKind.RLC})),
    "pinned_sheet_layout": ("layout", "pinned sheet layout", frozenset({OperandKind.PEC_SHEET})),
    "nonvacuum_masks": (0, "caller-supplied non-vacuum mask", frozenset({OperandKind.EPS, OperandKind.SIGMA, OperandKind.MU})),
    **{f"materials.{name}": (value, label, frozenset(kinds)) for name, value, label, kinds in (
        ("eps_r", 1, "permittivity", (OperandKind.EPS,)),
        ("sigma", 0, "conductivity", (OperandKind.SIGMA,)),
        ("mu_r", 1, "permeability", (OperandKind.MU,)),
        ("sigma_lumped.*", 0, "lumped conductivity", (OperandKind.PORT_STAMP, OperandKind.RLC)),
        ("eps_r_lumped.*", 0, "lumped permittivity", (OperandKind.PORT_STAMP, OperandKind.RLC)),
        ("mu_r_wire.*", 0, "wire permeability", (OperandKind.PEC_WIRE,)),
        ("edge_keep.*", 1, "PEC edge keep", (OperandKind.PEC_VOLUME,)),
        ("sigma_film.*", 0, "film conductivity", (OperandKind.THIN_FILM,)))},
    **{f"materials.components.{name}.*": (value, label, frozenset(kinds)) for name, value, label, kinds in (
        ("eps", 1, "component permittivity", (OperandKind.EPS,)),
        ("sigma", 0, "component conductivity", (OperandKind.SIGMA,)),
        ("mu", 1, "component permeability", (OperandKind.MU,)),
        ("eps_lumped", 0, "lumped permittivity", (OperandKind.PORT_STAMP, OperandKind.RLC)),
        ("sigma_lumped", 0, "lumped conductivity", (OperandKind.PORT_STAMP, OperandKind.RLC)),
        ("mu_wire", 0, "wire permeability", (OperandKind.PEC_WIRE,)),
        ("eps_update", 1, "component permittivity", (OperandKind.EPS,)),
        ("sigma_update", 0, "component conductivity", (OperandKind.SIGMA,)),
        ("mu_update", 1, "component permeability", (OperandKind.MU,)),
        ("upml_eps", 1, "component permittivity", (OperandKind.EPS,)),
        ("upml_sigma", 0, "component conductivity", (OperandKind.SIGMA,)))},
    **{f"materials.components.{kind}.{field}.*": (0, label, frozenset(kinds))
       for kind, label, kinds in (("debye", "Debye pole medium", (OperandKind.DEBYE,)),
                                  ("lorentz", "Drude/Lorentz pole medium", (OperandKind.LORENTZ, OperandKind.DRUDE)))
       for field in ("coefficients", "weights")},
    **{f"materials.components.tensor.{name}.*": (1, "tensor permittivity", frozenset({kind}))
       for name, kind in (("eps", OperandKind.EPS), ("inv_eps", OperandKind.EPS),
                          ("conformal_weights", OperandKind.PEC_VOLUME))},
    **{f"materials.components.tensor.node_smoothing.{name}*": (value, "node tensor", frozenset(kinds))
       for name, value, kinds in (("xi", "symmetric_identity", (OperandKind.EPS,)),
                                  ("triplets", "identity", (OperandKind.EPS,)),
                                  ("bulk", 1, (OperandKind.EPS,)), ("active", 0, (OperandKind.EPS,)),
                                  ("cross", 0, (OperandKind.EPS,)), ("pec_keep", 1, (OperandKind.PEC_VOLUME,)),
                                  ("guard_count", None, ()))},
    "debye_spec": ("spec", "Debye pole medium", frozenset({OperandKind.DEBYE})),
    "lorentz_spec": ("spec", "Drude/Lorentz pole medium", frozenset({OperandKind.LORENTZ, OperandKind.DRUDE})),
}


def material_rule(name):
    """Classify every spatial material leaf, including declaration-time views."""
    name = name.replace(".components.cells_view", "")
    for pattern, row in OPERANDS.items():
        if fnmatchcase(name, pattern):
            return row
    raise ValueError(f"Unclassified TFSF material operand: {name}")


# Values are (E/H family, component axis). These indices never name a pole, tensor entry,
# triplet or spatial axis. Builders: model/materials.py::realize_components,
# model/tensor_operands.py, geometry/node_tensor.py, model/node_pec.py,
# boundaries/pec.py, model/occupancy.py, sources/{sources,wire_radius}.py,
# model/thin_conductors.py, materials/thin_conductor.py and lumped.py.
COMPONENTS = {
    **{f"{family}.{c}": ("E", axis) for c, axis in enumerate("xyz") for family in (
        "pec_edge_masks", "aniso_eps", "aniso_inv_eps", "conformal_weights",
        *(f"materials.{name}" for name in (
            "sigma_lumped", "eps_r_lumped", "edge_keep", "sigma_film")),
        *(f"materials.components.{name}" for name in (
            "eps", "sigma", "eps_update", "sigma_update",
            "eps_lumped", "sigma_lumped", "upml_eps", "upml_sigma")),
        *(f"materials.components.tensor.{name}" for name in (
            "eps", "inv_eps", "conformal_weights")),
        *(f"materials.components.tensor.node_smoothing.{name}" for name in (
            "bulk", "active", "pec_keep")))},
    **{f"{family}.{c}": ("H", axis) for c, axis in enumerate("xyz") for family in (
        "materials.components.mu", "materials.components.mu_update",
        "materials.components.mu_wire", "materials.mu_r_wire")},
    **{f"sheet_impedance.mask_e{axis}": ("E", axis) for axis in "xyz"},
    **{f"{family}.e{axis}": ("E", axis) for family in ("lumped_rlc", "rlc_metas") for axis in "xyz"},
}

OPERAND_RULES: dict[str, Mapping[OperandKind, str]] = {}


@dataclass(frozen=True)
class OperandRecord:
    name: str
    kinds: frozenset[OperandKind]
    array: object
    vacuum: object
    label: str
    family: str | None
    component: str | None
    trailing: int
    source: object
    pole_spec: object = None


def _leaves(value, name):
    if value is None:
        return
    if hasattr(value, "shape"):
        yield name, value
    elif np.isscalar(value) and not isinstance(value, (str, bytes)):
        yield name, np.asarray(value)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _leaves(getattr(value, field.name), f"{name}.{field.name}")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        names = getattr(value, "_fields", range(len(value)))
        for key, child in zip(names, value):
            yield from _leaves(child, f"{name}.{key}")


def _record(name, kinds, array, vacuum, label, source=None, pole_spec=None):
    family, component = COMPONENTS.get(name.replace(".components.cells_view", ""), (None, None))
    return OperandRecord(name, kinds, array, vacuum, label, family, component,
                         2 if vacuum == "identity" else 0, source, pole_spec)


def realized_operands(grid, materials, updates, localized) -> tuple[OperandRecord, ...]:
    """Enumerate actual spatial operands without concretizing their values.

    ``source`` is the original update value, or None for material leaves.
    Assembly specs keep their tuple and masks by identity; setup ADE records
    retain their prebuilt source and have no assembly ``pole_spec``.
    """
    records = []
    for name, array in _leaves(materials, "materials"):
        if array.ndim < 3 and name not in ("materials.eps_r", "materials.sigma", "materials.mu_r"):
            continue
        vacuum, label, kinds = material_rule(name)
        if vacuum == "symmetric_identity":
            # Symmetric tensor order: xx, yy, zz, xy, xz, yz.
            vacuum = int(int(name.rsplit(".", 1)[1]) < 3)
        if vacuum is not None:
            records.append(_record(name, kinds, array, vacuum, label))
    for name, value in (*updates.items(), *localized):
        vacuum, label, kinds = OPERANDS[name]
        if value is None or vacuum is None or vacuum == "realized":
            continue
        source = value
        if vacuum == "spec":
            import jax.numpy as jnp
            poles, masks = value
            for index, _ in enumerate(poles):
                mask = None if masks is None else masks[index]
                array = jnp.asarray(True) if mask is None else mask
                records.append(_record(f"{name}.{index}", kinds, array, 0, label, source, value))
            continue
        if vacuum == "poles":
            coeffs = value[0]
            # Setups receive ADE coefficients, whose masked recurrence terms
            # carry the realized pole reach (not the initial, zero P state).
            names = ("alpha", "beta") if name == "debye" else ("a", "b", "c")
            value = {key: getattr(coeffs, key) for key in names}
            vacuum = 0
        elif vacuum == "sheet":
            value = {f"mask_e{a}": getattr(value, f"mask_e{a}") for a in "xyz"}
            vacuum = 0
        elif vacuum == "cells":
            import jax.numpy as jnp
            masks = {}
            for meta in value:
                mask = masks.get(meta.component, jnp.zeros(grid.shape, dtype=bool))
                masks[meta.component] = mask.at[meta.i, meta.j, meta.k].set(True)
            value, vacuum = masks, 0
        elif vacuum in ("window", "layout"):
            import jax.numpy as jnp
            mask = jnp.zeros(grid.shape, dtype=bool)
            if vacuum == "window":
                mask = mask.at[tuple(slice(*value.bounds[2*a:2*a+2]) for a in range(3))].set(True)
            else:
                from rfx.model.occupancy import layout_edge_indices
                for _, sheet, pair in value:
                    for tangent, values in enumerate(pair):
                        _, indices = layout_edge_indices(grid, sheet, tangent)
                        mask = mask.at[indices].max(values != 0)
            value, vacuum = mask, 0
        for operand, array in _leaves(value, name):
            records.append(_record(operand, kinds, array, vacuum, label, source))
    return tuple(records)


def operands_at(records, select) -> tuple[tuple[OperandRecord, object], ...]:
    """Return selected cuts, preserving broadcast axes and trailing tensors.

    ``select(record)`` returns three spatial slices, or None to skip a record.
    Scalars and arrays with fewer than three spatial dimensions remain whole.
    No vacuum comparison is performed, so cuts may carry tracers.
    """
    result = []
    for record in records:
        spatial_slice = select(record)
        if spatial_slice is None:
            continue
        array = record.array
        start = array.ndim - 3 - record.trailing
        if start < 0:
            cut = array
        else:
            slices = [slice(None)] * array.ndim
            for a, sl in enumerate(spatial_slice):
                slices[start + a] = slice(None) if array.shape[start + a] == 1 else sl
            cut = array[tuple(slices)]
        result.append((record, cut))
    return tuple(result)
