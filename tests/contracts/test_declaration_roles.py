"""Step B registry completeness against declarations, independent of its rows."""
from dataclasses import is_dataclass

from rfx import Box, Cylinder, OrientedBox, Sphere
from rfx.farfield import NTFFBox
from rfx.model import declaration_roles as D
from rfx.sources.sources import GaussianPulse, ModulatedGaussian, CWSource, CustomWaveform
from tests.contracts import path_disposition as T
from tests.contracts.test_declaration_transforms import VALUE_CASES
from tests.unit.runners import test_path_disposition_cells as B


def reachable_records():
    records = {}

    def walk(value):
        cls = type(value)
        if is_dataclass(value) or hasattr(cls, "_fields"):
            records[cls] = D.record_fields(cls)
            for name in records[cls]:
                walk(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)

    for spec in B.FEATURES.values():
        sim = B._build(spec, "fwd_uniform", True)
        for name in T.TABLE:
            if name in vars(sim):
                walk(vars(sim)[name])
    # Public shape/waveform alternatives to the inventory's chosen builders.
    sim = B._base("fwd_uniform", source=None)
    for shape in (Sphere((.006,) * 3, .002), Cylinder((.006,) * 3, .002, .004),
                  OrientedBox((.006,) * 3, (.002,) * 3)):
        sim.add(shape, material="pec")
    for waveform in (GaussianPulse(5e9), ModulatedGaussian(5e9), CWSource(5e9), CustomWaveform(lambda t: t)):
        sim.add_source((.004, .006, .006), "ez", waveform=waveform)
    walk(sim._geometry)
    walk(sim._ports)
    # NTFF's tuple declaration is converted to this record before scan.
    records[NTFFBox] = NTFFBox._fields
    return records


def test_reachable_fields_have_exactly_one_role():
    reachable = reachable_records()
    missing = [(cls.__name__, field) for cls, names in reachable.items()
               for field in names if (cls, field) not in D.ROLES]
    assert not missing, f"unassigned declaration fields: {missing}"
    outside = [(cls.__name__, field) for cls, field in D.ROLES
               if cls not in reachable or field not in reachable[cls]]
    assert not outside, f"roles outside reachable declarations: {outside}"
    assert set(D.ROLES.values()) <= {D.VALUE, D.STRUCTURE, D.STRUCTURE_TRACEABLE}
    for field in ("_dx_profile", "_dy_profile", "_dz_profile"):
        assert D.SIMULATION_ROLES[field] == D.VALUE
    assert D.SIMULATION_ROLES["_nominal_profiles"] == D.STRUCTURE


def test_every_value_field_has_transform_witness():
    covered = {(record, field) for record, field, *_ in VALUE_CASES}
    required = {(cls.__name__, field) for (cls, field), role in D.ROLES.items() if role == D.VALUE}
    required |= {("Simulation", field) for field, role in D.SIMULATION_ROLES.items() if role == D.VALUE}
    assert required == covered, (required - covered, covered - required)
