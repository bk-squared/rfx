"""Every material-aware H builder consumes component mu or refuses its record."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core import yee
from rfx.grid import Grid
from rfx.runners import _distributed_common as dc


@pytest.mark.parametrize("periodic", [(False,)*3, (True, False, True)])
def test_no_record_keeps_the_exact_cell_objects(periodic):
    mats = yee.init_materials((4,)*3)
    mu = jnp.arange(64, dtype=jnp.float32).reshape((4,)*3)+1
    mats = mats._replace(mu_r=mu)
    parts = yee.component_h_materials(mats, periodic=periodic)
    assert all(part is mu for part in parts)


def _fixture():
    grid = Grid(freq_max=10e9, domain=(.006,)*3, dx=.001, cpml_layers=2)
    mats = yee.init_materials(grid.shape)
    record = tuple(jnp.full(grid.shape, a) for a in (1., 3., 7.))
    marked = mats._replace(mu_r_wire=record)
    xyz = jnp.indices(grid.shape, dtype=jnp.float32)
    state = yee.init_state(grid.shape)._replace(
        ex=xyz[1]+2*xyz[2], ey=3*xyz[0]+xyz[2], ez=xyz[0]+4*xyz[1])
    return grid, mats, marked, state


@pytest.mark.parametrize("path", ["uniform", "nonuniform", "fast", "distributed_v2",
                                   "distributed_nu", "cpml_uniform", "cpml_nonuniform"])
def test_h_builder_carries_component_record(path):
    from rfx.boundaries.cpml import apply_cpml_h, init_cpml
    grid, mats, marked, state = _fixture()
    inv = tuple(jnp.full(n, 1/grid.dx) for n in grid.shape)

    def build(m):
        if path == "uniform":
            return yee.update_h(state, m, grid.dt, grid.dx)
        if path == "nonuniform":
            return yee.update_h_nu(state, m, grid.dt, *inv)
        if path == "fast":
            return yee.update_h_fast(state, yee.precompute_coeffs(m, grid.dt, grid.dx).ch)
        if path == "distributed_v2":
            return dc._update_h_local(state, m, grid.dt, grid.dx)
        if path == "distributed_nu":
            return dc._update_h_local_nu(state, m, grid.dt, *inv, *inv)
        # The two public lanes share the CPML H builder; exercise both grids.
        g = grid
        if path == "cpml_nonuniform":
            from rfx import Simulation
            g = Simulation(freq_max=10e9, domain=(.006,)*3, dx=.001,
                           dx_profile=np.full(6, .001), cpml_layers=2)._build_nonuniform_grid()
        params, psi = init_cpml(g)
        return apply_cpml_h(state, params, psi, g, materials=m)[0]

    with jax.disable_jit():
        original = build(mats)
        corrected = build(marked)
    for i, factor in enumerate((2., 4., 8.)):
        before, after = np.asarray(original[i+3]), np.asarray(corrected[i+3])
        assert np.max(abs(before)) > 0
        np.testing.assert_allclose(after, before/factor, rtol=3e-7, atol=1e-12)


@pytest.mark.parametrize("path", ["split", "shard", "distributed_nu", "distributed_h_shard",
                                   "subgridded", "subgridded_reference", "sat_h", "sat_e", "upml", "adi"])
def test_unimplemented_carrier_refuses_non_none_record(path):
    from rfx import Simulation
    from rfx.boundaries.upml import init_upml
    from rfx.runners.distributed_nu import run_nonuniform_distributed_pec
    from rfx.runners.distributed_v2 import _shard_materials
    from rfx.subgridding import jit_runner as sg
    grid, _, marked, _ = _fixture()
    with pytest.raises(NotImplementedError, match="radius"):
        if path == "split":
            dc._split_materials(marked, 2)
        elif path == "shard":
            _shard_materials(marked, None)
        elif path == "distributed_nu":
            run_nonuniform_distributed_pec(None, marked, None, 1, n_devices=2)
        elif path == "distributed_h_shard":
            dc.update_h_nu_shmap(None, marked, None, 1., *([None]*6))
        elif path == "subgridded":
            sg.run_subgridded_jit(None, marked, marked, None, 1)
        elif path == "subgridded_reference":
            from rfx.subgridding.sbp_sat_3d import step_subgrid_3d
            step_subgrid_3d(None, None, mats_c=marked)
        elif path in ("sat_h", "sat_e"):
            fn = getattr(sg, f"_z_slab_material_coupling_{path[-1]}_3d")
            fn(None, None, marked, marked, None)
        elif path == "upml":
            init_upml(grid, marked)
        elif path == "adi":
            sim = Simulation(freq_max=10e9, domain=(.006,)*3, dx=.001, solver="adi")
            sim._run_adi_from_materials(grid, marked, None, None, n_steps=1, lane="run_adi")


def test_h_builder_inventory_reads_the_shared_owner():
    """Structural backstop for the vacuum-only and refusing coefficient sites."""
    from rfx import adi
    from rfx.boundaries import cpml, upml
    from rfx.runners import distributed_nu
    from rfx.subgridding import jit_runner as sg, sbp_sat_1d, sbp_sat_2d, disjoint_3d
    from rfx.sources import tfsf, tfsf_2d, tfsf_oblique_open, waveguide_port, msl_port, coaxial_port
    sites = [yee.update_h, yee.update_h_nu, yee.precompute_coeffs,
             dc._update_h_local, dc._update_h_local_nu, cpml.apply_cpml_h,
             dc._apply_cpml_h_distributed, distributed_nu._apply_cpml_h_local_nu,
             upml.init_upml, adi.adi_step_2d, adi.adi_step_3d, adi.apply_adi_cpml_2d,
             sg._z_slab_material_coupling_h_3d, sg._z_slab_material_coupling_e_3d,
             dc.cpml_coeff_h_vacuum, sbp_sat_1d._update_h_1d,
             sbp_sat_2d._update_hx_2d, sbp_sat_2d._update_hy_2d,
             tfsf.update_tfsf_1d_h, tfsf.apply_tfsf_h, tfsf._apply_closed_box_h,
             tfsf_2d._update_h_tmz, tfsf_2d._update_h_tez, tfsf_2d.apply_tfsf_2d_h,
             tfsf_oblique_open.apply_methodB_h, waveguide_port.apply_waveguide_port_h,
             msl_port.make_msl_port_sources_jm,
             coaxial_port.build_coaxial_tem_plane_source_specs, disjoint_3d]
    for fn in sites:
        assert "component_h_materials(" in inspect.getsource(fn), getattr(fn, "__qualname__", fn.__name__)


def test_cpml_legacy_material_view_and_vacuum_fallback():
    from types import SimpleNamespace
    from rfx.boundaries.cpml import apply_cpml_h, init_cpml
    grid, mats, _, state = _fixture()
    params, psi = init_cpml(grid)
    for view, reference_view in ((SimpleNamespace(mu_r=mats.mu_r), mats),
                                 (SimpleNamespace(eps_r=mats.eps_r), None)):
        reference = apply_cpml_h(state, params, psi, grid, materials=reference_view)[0]
        result = apply_cpml_h(state, params, psi, grid, materials=view)[0]
        for a, b in zip(result[3:6], reference[3:6]):
            np.testing.assert_array_equal(a, b)
