"""Axis C coverage and the shared operand query's assembly/setup interface."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Import each rule owner so its table is registered even in an isolated run.
from rfx.boundaries import features, tfsf
from rfx.model.operands import (
    COMPONENTS, OPERANDS, OPERAND_RULES, OperandKind, OperandRecord,
    material_rule, operands_at, realized_operands,
)


def test_every_rule_names_every_kind():
    assert "tfsf_injection_planes" in OPERAND_RULES
    assert "waveguide_port_section" in OPERAND_RULES
    for name, table in OPERAND_RULES.items():
        assert set(table) == set(OperandKind), name
        allowed = {"accept", "refuse", "medium"} if name == "waveguide_port_section" else {"accept", "refuse"}
        assert set(table.values()) <= allowed, name


def test_every_operand_row_has_kinds():
    containers = {"materials"}
    assert {name for name, (vacuum, _, _) in OPERANDS.items() if vacuum == "leaves"} == containers
    for name, (vacuum, _, kinds) in OPERANDS.items():
        assert isinstance(kinds, frozenset), name
        assert kinds <= set(OperandKind), name
        if vacuum is not None and name not in containers:
            assert kinds, name
        elif name in ("sources", "mag_sources"):
            assert kinds == {OperandKind.SOURCE_STAMP}, name
        else:
            assert not kinds, name


def test_aliases_and_tfsf_policy():
    assert tfsf.INJECTION_OPERANDS is OPERANDS
    assert tfsf.injection_material_rule is material_rule
    assert tfsf._INJECTION_COMPONENTS is COMPONENTS
    table = OPERAND_RULES["tfsf_injection_planes"]
    assert {kind for kind, decision in table.items() if decision == "accept"} == {OperandKind.SOURCE_STAMP}
    assert material_rule("materials.components.tensor.conformal_weights.0")[2] == {OperandKind.PEC_VOLUME}
    assert material_rule("materials.components.tensor.node_smoothing.pec_keep.0")[2] == {OperandKind.PEC_VOLUME}
    assert material_rule("materials.components.tensor.node_smoothing.guard_count")[2] == frozenset()
    with pytest.raises(ValueError, match="^Unclassified TFSF material operand: materials.unknown$"):
        material_rule("materials.unknown")


@pytest.mark.parametrize("shape,trailing,expected", [
    ((), 0, ()), ((4, 5), 0, (4, 5)),
    ((4, 5, 6), 0, (2, 2, 2)),
    ((2, 4, 5, 6), 0, (2, 2, 2, 2)),
    ((2, 1, 5, 1), 0, (2, 1, 2, 1)),
    ((4, 5, 6, 3, 3), 2, (2, 2, 2, 3, 3)),
    ((1, 5, 1, 3, 3), 2, (1, 2, 1, 3, 3)),
])
def test_query_spatial_axes(shape, trailing, expected):
    array = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    record = OperandRecord("operand", frozenset({OperandKind.EPS}), array,
                           "identity" if trailing else 1, "material", None, None, trailing, None)
    slices = (slice(1, 3), slice(2, 4), slice(3, 5))
    ((found, cut),) = operands_at((record,), lambda _: slices)
    assert found is record
    assert cut.shape == expected
    if shape == (4, 5, 6):
        np.testing.assert_array_equal(cut, array[1:3, 2:4, 3:5])
    if len(shape) - trailing < 3:
        assert cut is array
    assert operands_at((record,), lambda _: None) == ()


def test_assembly_pole_specs_preserve_identity_and_traced_masks():
    from rfx import Box, Simulation
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import LorentzPole
    from rfx.model.materials import assemble_cells

    sim = Simulation(freq_max=10e9, domain=(.008, .008, .008), dx=.001,
                     boundary="pec")
    sim.add_material("poles", debye_poles=[DebyePole(1., 1e-10)],
                     lorentz_poles=[LorentzPole(2*np.pi*5e9, 1e10, (2*np.pi*5e9)**2)])
    sim.add(Box((.002, .002, .002), (.006, .006, .006)), material="poles")
    grid = sim._build_grid()
    materials, debye, lorentz, *_ = assemble_cells(sim, grid, pec_sheets=[], pec_wires=[])
    updates = {"debye_spec": debye, "lorentz_spec": lorentz}
    records = realized_operands(grid, materials, updates, ())
    leaves = [record for record in records if record.pole_spec is None]
    assert leaves and all(record.source is None for record in leaves)
    for name, spec in updates.items():
        poles = [record for record in records if record.name.startswith(name + ".")]
        assert len(poles) == len(spec[0]) == 1
        for index, record in enumerate(poles):
            assert record.pole_spec is spec
            assert record.source is spec
            assert record.array is spec[1][index]
            assert record.vacuum == 0
            assert record.kinds == OPERANDS[name][2]

    @jax.jit
    def query(mask):
        specs = {name: (spec[0], [mask]) for name, spec in updates.items()}
        traced = realized_operands(grid, None, specs, ())
        assert len(traced) == 2
        for record, spec in zip(traced, specs.values()):
            assert record.pole_spec is spec
            assert record.source is spec
            assert record.array is mask
            assert isinstance(record.array, jax.core.Tracer)
        cuts = operands_at(traced, lambda _: (slice(2, 4),) * 3)
        assert all(isinstance(cut, jax.core.Tracer) for _, cut in cuts)
        return cuts[0][1] | cuts[1][1]

    mask = debye[1][0]
    np.testing.assert_array_equal(query(mask), mask[2:4, 2:4, 2:4])


@pytest.mark.parametrize("masks", [None, [None, None]])
def test_spec_unmasked_poles_and_absent_specs(masks):
    spec = ([object(), object()], masks)
    records = realized_operands(None, None, {"debye_spec": spec, "lorentz_spec": None}, ())
    assert len(records) == 2
    for record in records:
        assert record.pole_spec is spec
        assert record.array.shape == ()
        assert bool(record.array)
        assert record.vacuum == 0


@pytest.mark.parametrize("name,coefficients", [("debye", ("alpha", "beta")), ("lorentz", ("a", "b", "c"))])
def test_prebuilt_poles_keep_source_without_spec(name, coefficients):
    arrays = {key: jnp.ones((2, 1, 3, 4)) for key in coefficients}
    source = (SimpleNamespace(**arrays), object())
    records = realized_operands(None, None, {name: source}, ())
    assert [record.name for record in records] == [f"{name}.{key}" for key in coefficients]
    for record, array in zip(records, arrays.values()):
        assert record.source is source
        assert record.pole_spec is None
        assert record.array is array
        assert record.kinds == OPERANDS[name][2]


def test_material_components_and_skipped_adapters():
    arrays = (jnp.ones((3, 4, 5)),) * 3
    materials = {"components": {"cells_view": {"eps_r": arrays[0]}, "mu_wire": arrays,
                                "tensor": {"node_smoothing": {"triplets": arrays[0][..., None, None]}}}}
    records = realized_operands(None, materials, {"sources": object(), "mag_sources": object(),
                                                "pec_sheets": object(), "pec_wires": object()}, ())
    assert len(records) == 5
    assert records[0].kinds == {OperandKind.EPS}
    assert [(r.family, r.component) for r in records[1:4]] == [("H", a) for a in "xyz"]
    assert records[-1].trailing == 2
