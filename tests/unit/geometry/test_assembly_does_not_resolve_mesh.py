"""Material assembly reads the declared mesh, never the resolved view (#1070 follow-up).

``Simulation._domain`` is a ``_MeshField`` descriptor: reading it runs
``_resolve_mesh``. ``differentiable_material_fit.forward`` builds the grid once
from a template simulation and then, per iteration, constructs a FRESH
simulation carrying traced materials and hands it that grid through
``_assemble_materials``. The fresh simulation has never resolved its mesh, so
any descriptor read inside assembly runs the auto-mesh planner on a tracer and
raises. PR #1136 added a ``self._domain`` read there and the GPU suite went red
on ``test_recover_debye_reference_mode_public_entry`` (VESSL 369367262302 on
a6d6fce1). Assembly is handed a built grid and must not plan a mesh; this pins
the invariant in seconds, without the nine-minute fit and without a trace.
"""
import pytest

from rfx import Box, Simulation
from rfx.api import _mesh
from rfx.boundaries.spec import Boundary, BoundarySpec


def _conformal_walls():
    return BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec", conformal=True), z="cpml")


def _i580_like_sim(boundary=None):
    # Same declaration as the fit's fixture: auto (dx-less) uniform mesh, a
    # dielectric slab spanning the transverse domain. ``boundary`` selects the
    # conformal-wall branch of the assembly, which reads the domain too.
    kwargs = {} if boundary is None else {"boundary": boundary()}
    sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009), **kwargs)
    sim.add_material("dut", eps_r=2.0, sigma=0.0)
    sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
    return sim


@pytest.mark.parametrize("boundary", [None, _conformal_walls],
                         ids=["default-boundary", "conformal-pec-walls"])
def test_assembly_of_a_fresh_sim_never_resolves_its_mesh(monkeypatch, boundary):
    grid = _i580_like_sim(boundary)._build_grid()   # the template's one legitimate resolution
    fresh = _i580_like_sim(boundary)                # never resolved, like forward()'s per-step sim

    original_get = _mesh._MeshField.__get__

    def _refuse(self, obj, owner=None):
        if obj is not None:
            raise AssertionError(
                f"_assemble_materials read the resolved mesh field {self.name!r} on a "
                f"simulation that had not resolved its mesh; it must use the grid it "
                f"was given or _declared_mesh")
        return original_get(self, obj, owner)

    monkeypatch.setattr(_mesh._MeshField, "__get__", _refuse)
    fresh._assemble_materials(grid)


def test_after_a_freeze_assembly_sees_the_frozen_domain_not_the_callers_container():
    """``_freeze_mesh`` owns a snapshot so a caller-owned list mutated afterwards
    cannot move the model; the unresolved read must honour it (review of #1140)."""
    dom = [0.024, 0.009, 0.009]
    sim = Simulation(freq_max=5e9, domain=dom, dx=1.5e-3)
    sim.freeze_mesh()
    dom[2] = 0.050
    assert tuple(sim._unresolved_domain) == (0.024, 0.009, 0.009)
    assert tuple(sim._unresolved_domain) == tuple(sim._domain)

