"""Small declared structures for the M3 diagnostic contracts (no field oracle)."""
from dataclasses import replace

import numpy as np

from rfx import Box, Simulation


U = 2.0**-13


def structure(graded=False, *, name='checked', offset=3, height=2 * U, feed=4 * U, freq_max=20e9, length=32 * U):
    sim = Simulation(
        freq_max=freq_max,
        domain=(length, 12 * U, 10 * U),
        dx=U,
        boundary="cpml",
        cpml_layers=2,
        snap="declared",
        **({"dz_profile": np.full(10, U)} if graded else {}),
    )
    sim.add_material(
        "substrate",
        eps_r=3.66,
    )
    sim.add(
        Box((0, 0, 0), (length, 12 * U, height)),
        material="substrate",
    )
    sim.add(
        Box((0, 0, 0), (length, 12 * U, 0)),
        material="pec",
    )
    sim.add(
        Box((feed, 4 * U, height), (length, 8 * U, height)),
        material="pec",
    )
    sim.add_msl_port(
        (feed, 6 * U, 0),
        width=4 * U,
        height=height,
        direction="+x",
        name=name,
        mode="uniform",
        eps_r_sub=3.66,
        n_probe_offset=offset,
        n_probe_spacing=2,
        n_probes=3,
    )
    return sim


def family_warnings(sim):
    import warnings
    try:
        grid = sim._build_realized_grid()
    except ValueError:
        grid = None
    low = [getattr(grid, f'pad_{axis}_lo', 2) * U for axis in 'xyz']
    high = [getattr(grid, f'pad_{axis}_hi', 2) * U for axis in 'xyz']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sim._check_msl_port_geometry(U, low, high)
    return [w.message for w in caught if getattr(w.message, 'code', '').startswith('msl_port')]


def cases(graded=False):
    """Regular setups plus explicit unavailable-dependency fault injection."""
    yield 'base', structure(graded)
    yield (
        "alignment",
        structure(
            graded,
            height=2.25 * U,
        ),
    )
    yield (
        "gap_difference",
        structure(
            graded,
            height=4.45 * U,
        ),
    )
    yield (
        "clamped",
        structure(
            graded,
            offset=40,
        ),
    )
    sim = structure(
        graded,
        offset=40,
    )
    sim._build_realized_grid = lambda: (_ for _ in ()).throw(ValueError('grid unavailable'))
    yield 'unavailable_grid_extrapolation', sim
    yield (
        "near_absorber",
        structure(
            graded,
            offset=23,
        ),
    )
    sim = structure(graded)
    sim.add(
        Box((10 * U, 4 * U, 0), (11 * U, 8 * U, 2 * U)),
        material="pec",
    )
    yield 'reflector', sim
    sim = structure(
        graded,
        offset=None,
    )
    sim.add(
        Box((10 * U, 4 * U, 0), (11 * U, 8 * U, 2 * U)),
        material="pec",
    )
    yield 'automatic_placement', sim
    sim = structure(
        graded,
        offset=10,
    )
    sim._msl_ports.append(
        replace(
            sim._msl_ports[0],
            name="opposite",
            position=(12 * U, 6 * U, 0),
            direction="-x",
        )
    )
    yield 'cross_feed', sim
    sim = structure(graded)
    sim._msl_assemble_once = lambda: None
    yield 'assembly_unavailable', sim
    sim = structure(graded)
    sim._msl_conductor_gap = lambda *a: (_ for _ in ()).throw(ValueError('attachment unavailable'))
    yield 'attachment_unavailable', sim
    sim = structure(graded)
    sim._resolve_msl_probe_entries = lambda *a: (_ for _ in ()).throw(ValueError('placement unavailable'))
    yield 'placement_unavailable', sim
    sim = structure(graded)
    def placement(grid, sim=sim):
        import warnings
        warnings.warn('placement detail')
        return sim._msl_ports
    sim._resolve_msl_probe_entries = placement
    yield 'placement_note', sim


def captured_cases(graded=False):
    """Every warning site, including failed inspection dependencies, on both grids."""
    import warnings
    from unittest.mock import patch
    import rfx.preflight.msl as msl
    import rfx.preflight.line_stub as stub
    results = {}
    for key, sim in cases(graded):
        results[key] = family_warnings(sim)
    sim = structure(graded)
    grid = sim._build_realized_grid()
    record = msl.msl_probe_clearance_for_port(sim, sim._msl_ports[0], grid)
    unavailable = replace(
        record,
        status="unavailable",
        note="no probe metadata",
        unevaluated_conductors=("unsupported conductor",),
    )
    with patch.object(
        msl,
        "msl_probe_clearance_for_port",
        return_value=unavailable,
    ):
        results["incomplete_scan"] = family_warnings(sim)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            msl.preflight_msl_probe_clearance(sim, warnings)
        results["calculator_unavailable"] = [w.message for w in caught]
    with patch.object(
        msl,
        "msl_probe_clearance_for_port",
        side_effect=ValueError("scan unavailable"),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            msl.preflight_msl_probe_clearance(sim, warnings)
        results["calculator_scan_failed"] = [w.message for w in caught]
    # The same physical distance/threshold condition as general preflight.
    sim.add(
        Box((10 * U, 4 * U, 0), (11 * U, 8 * U, 2 * U)),
        material="pec",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        msl.preflight_msl_probe_clearance(sim, warnings)
    results['calculator_clearance'] = [w.message for w in caught]
    for key, error in [('stub_realization', ValueError('conductor realization failed')),
                       ('stub_inspection', NotImplementedError('unsupported conductor'))]:
        with patch.object(
            stub,
            "line_stub_findings",
            side_effect=error,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                stub.preflight_line_stubs(sim, warnings)
        results[key] = [w.message for w in caught]
    # Physical open tail, large enough to be resonant in this read interval.
    sim = structure(
        graded,
        feed=16 * U,
    )
    sim._geometry[-1] = replace(
        sim._geometry[-1],
        shape=Box((0, 4 * U, 2 * U), (32 * U, 8 * U, 2 * U)),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        stub.preflight_line_stubs(sim, warnings)
    results['stub_behind_port'] = [w.message for w in caught]
    return results
