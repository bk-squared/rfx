"""Every declaration surface participates in both diagnostic cache keys."""
import gc
import weakref

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api import _declarations as declarations


POINT = (0.005, 0.005, 0.005)
CALLS = {
    "add": ((Box((0.002,)*3, (0.004,)*3),), {"material": "pec"}),
    "add_material": (("board",), {"eps_r": 3.4}),
    "add_source": ((POINT,), {}),
    "add_polarized_source": ((POINT,), {"polarization": "ex"}),
    "add_coaxial_port": ((POINT,), {}),
    "add_thin_conductor": ((Box((0.002, 0.002, 0.004), (0.008, 0.008, 0.004)),), {}),
    "add_pinned_sheet": ((), {"plane_index": 3, "i_range": (2, 6), "j_range": (2, 6)}),
    "add_port": ((POINT,), {}),
    "add_msl_port": ((POINT,), {"width": 0.002, "height": 0.002}),
    "add_lumped_rlc": ((POINT,), {"R": 50, "topology": "parallel"}),
    "add_waveguide_port": ((0.003,), {"probe_offset": 2, "ref_offset": 1}),
    "add_floquet_port": ((0.003,), {}),
    "add_probe": ((POINT,), {}),
    "add_vector_probe": ((POINT,), {}),
    "add_dft_plane_probe": ((), {"axis": "z", "coordinate": 0.005, "region": (2, 4, 2, 4)}),
    "add_flux_monitor": ((), {"axis": "z", "coordinate": 0.005}),
    "add_ntff_box": (((0.002,)*3, (0.008,)*3), {}),
    "add_current_moment_monitor": (((0.002,)*3, (0.008,)*3, 0.002, np.array([1e9])), {}),
    "add_refinement": (((0.003, 0.007),), {"ratio": 2, "validation": "research"}),
    "add_tfsf_source": ((), {"closed_box": True}),
}


def model(**kwargs):
    return Simulation(freq_max=10e9, domain=(0.012,)*3, dx=0.001,
                      boundary="cpml", cpml_layers=2, snap="declared", **kwargs)


def declare(sim, name):
    args, kwargs = CALLS[name]
    getattr(sim, name)(*args, **kwargs)


def test_declaration_rich_attributes_are_classified():
    declared = dict(declarations.DECLARATIONS)
    excluded = dict(declarations.NOT_DECLARATIONS)
    assert len(declared) == len(declarations.DECLARATIONS)
    assert not declared.keys() & excluded.keys()
    methods = {name for name in dir(Simulation) if name.startswith(("add_", "set_"))}
    assert methods == CALLS.keys() - {"add"}
    # TF/SF and port declarations are mutually exclusive. Exercise every call
    # on its own valid builder, and accumulate all compatible calls as well.
    rich = model()
    attributes = set(vars(rich))
    for name in CALLS:
        sim = model()
        declare(sim, name)
        attributes.update(vars(sim))
        if name not in {"add_tfsf_source", "add_floquet_port", "add_waveguide_port"}:
            declare(rich, name)
    rich.freeze_mesh()
    attributes.update(vars(rich))
    assert attributes <= declared.keys() | excluded.keys(), attributes - declared.keys() - excluded.keys()


@pytest.mark.parametrize("name", CALLS)
def test_each_declaration_changes_key(name):
    sim = model()
    before = declarations.declaration_key(sim)
    declare(sim, name)
    after = declarations.declaration_key(sim)
    assert before != after, name
    assert after == declarations.declaration_key(sim)
    assert hash(after) == hash(declarations.declaration_key(sim))


@pytest.mark.parametrize("axis", "xyz")
def test_in_place_profile_edit_misses_both_caches(axis):
    profile = np.full(12, 0.001)
    sim = model(**{f"d{axis}_profile": profile})
    before = sim._campaign_ctx()
    record = sim.realized_geometry()
    nodes = before.nodes["xyz".index(axis)].copy()
    assert sim._campaign_ctx() is before
    assert sim.realized_geometry() is record
    profile[4] += 0.0001
    profile[5] -= 0.0001
    after = sim._campaign_ctx()
    assert after is not before
    assert sim.realized_geometry() is not record
    assert not np.array_equal(after.nodes["xyz".index(axis)], nodes)


@pytest.mark.parametrize("surface", ["_geometry", "_materials"])
def test_key_retains_replaced_entry_across_gc(surface):
    sim = model()
    declare(sim, "add" if surface == "_geometry" else "add_material")
    entry = sim._geometry[0] if surface == "_geometry" else sim._materials["board"]
    reference = weakref.ref(entry)
    before = declarations.declaration_key(sim)
    del entry
    if surface == "_geometry":
        sim._geometry.clear()
    else:
        sim._materials.clear()
    gc.collect()
    assert reference() is not None  # the key, not an allocator coincidence, owns it
    declare(sim, "add" if surface == "_geometry" else "add_material")
    assert declarations.declaration_key(sim) != before
    del before
    gc.collect()
    assert reference() is None


def test_traced_profile_key_uses_identity_without_concretization():
    def traced(profile):
        sim = model(dz_profile=profile)
        key = declarations.declaration_key(sim)
        assert key == declarations.declaration_key(sim)
        assert hash(key) == hash(declarations.declaration_key(sim))
        sim._dz_profile = profile * 1.0
        assert key != declarations.declaration_key(sim)
        return profile.sum()
    jax.make_jaxpr(traced)(np.full(12, 0.001))


def test_array_shape_dtype_and_contents_are_keyed():
    sim = model(dz_profile=np.full(12, 0.001, dtype=np.float32))
    before = declarations.declaration_key(sim)
    sim._dz_profile = sim._dz_profile.copy()
    assert declarations.declaration_key(sim) == before
    sim._dz_profile = sim._dz_profile.astype(np.float64)
    assert declarations.declaration_key(sim) != before
    before = declarations.declaration_key(sim)
    sim._dz_profile = sim._dz_profile.reshape(3, 4)
    assert declarations.declaration_key(sim) != before
