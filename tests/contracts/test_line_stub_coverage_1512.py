"""Always-on build gate for the converted #1512 measurement fixtures."""
import numpy as np
import pytest

from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.geometry.port_termination import default_msl_terminates
from rfx.preflight.line_stub import line_stub_findings
from rfx.probes.msl_wave_decomp import realized_trace_planes_on_column
from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry


def converted_builder(case):
    if case == "uniform":
        from tests.locks.test_patch_edgefed_s11_passivity import _build_patch_sim
        return _build_patch_sim()
    if case == "nu":
        from tests.locks.test_msl_nu_sparam_gate import _build_patch_sim_nu
        return _build_patch_sim_nu()
    if case.startswith("broad-"):
        from scripts.diagnostics.run_msl_broad_e5_sweep import (
            build_simulation, case_geometry_params, enumerate_cases,
        )
        spec = next(spec for spec in enumerate_cases() if spec.case_id == case[6:])
        return build_simulation(spec, case_geometry_params(spec))
    from tests.locks.test_patch_edgefed_resonance_harminv import _build
    return _build(True)[0]


@pytest.fixture
def _release_case_memory():
    """Each case builds a fine board; without this the file holds them all
    (4.3 GB for 15 cases against 0.7-1.4 GB each), which tipped the 14-worker
    contract gate over its 32 GiB limit (2026-10-07)."""
    yield
    import gc
    import jax
    jax.clear_caches()   # measured: 2.8 GB with it, 3.7 GB with gc alone
    gc.collect()


@pytest.mark.parametrize("case", ["uniform", "nu", "harminv-fed", *[
    f"broad-{substrate}-{band}-{geometry}-{resolution}"
    for substrate, band, resolution in [
        ("ro4003c", "high", "sub4"), ("ro4003c", "high", "sub6"),
        ("ro4003c", "low", "sub4"), ("teflon", "high", "sub4"),
        ("ro4003c", "low", "sub6"), ("teflon", "high", "sub6"),
    ] for geometry in ("thru", "open_stub")
]])
def test_converted_fixture_port_coverage(case, _release_case_memory):
    sim = converted_builder(case)
    grid = sim._build_realized_grid()
    report = sim.preflight()
    assert not report.errors, [str(error) for error in report.errors]
    assert not line_stub_findings(sim, grid), "one-cell allowance must be silent"
    assert not any(f.code == "line_stub_behind_port" for f in report)
    sheets, wires, lossy = [], [], []
    assemble = sim._assemble_materials_nu if hasattr(grid, "dx_arr") else sim._assemble_materials
    result = assemble(grid, pec_sheets=sheets, pec_wires=wires, sheet_specs=lossy)
    edges = realized_pec_edge_masks(result[3], sheets, wires, periodic=sim._periodic_flags())
    for pe in sim._msl_ports:
        span = msl_cross_section_span(grid, msl_port_from_entry(pe))
        refs = default_msl_terminates(sim, grid, position=pe.position, width=pe.width,
                                      height=pe.height, direction=pe.direction)
        feed = sim._geometry[1] if case.startswith("broad-") else sim._thin_conductors[1]
        assert any(ref.entry is feed for ref in refs), "feed not detected"
        planes = realized_trace_planes_on_column(
            edges, 2, (span["i_feed"], span["w_centre"]), span["n_hi"],
            periodic=sim._periodic_flags())
        assert planes != (None, None), "extractor cannot find the trace at the port column"
        assert np.isfinite(planes).all()
        occupied = np.flatnonzero(np.asarray(edges[0])[:, span['w_centre'], span['n_hi']])
        endpoint = occupied[0] if pe.direction == '+x' else occupied[-1] + 1
        assert endpoint == span['i_feed'], "realized signal endpoint must equal the port node"
