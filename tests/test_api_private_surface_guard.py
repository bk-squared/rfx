"""B.8 import / private-surface regression guard.

Part B (`api.py` → `api/` package) plans to split the 7718-line module
into a package. ~79 test call-sites reach into `Simulation` private
methods (`_build_grid`, `_build_nonuniform_grid`, `_assemble_materials`)
and the public re-export surface (`from rfx.api import Simulation`).

This test pins that contract *before* any structural refactor so the
file→package conversion cannot silently drop a name or turn a method
into a free function. It is a fast, always-run guard (no `slow` /
`gpu` marker) — see roadmap §B.8 and the Stage 0 plan entry.
"""

import numpy as np


def test_rfx_api_module_imports():
    """`import rfx.api` and the documented re-export surface resolve."""
    import rfx.api  # noqa: F401
    from rfx.api import Simulation  # noqa: F401

    # B.8 external import surface — these names must stay importable
    # from `rfx.api` across the planned package conversion.
    from rfx.api import (  # noqa: F401
        AD_MemoryEstimate,
        CoaxialSMatrixResult,
        ForwardResult,
        MATERIAL_LIBRARY,
        MaterialSpec,
        MSLSMatrixResult,
        Result,
        WaveguideSMatrixResult,
        WaveguideSParamResult,
    )


def test_build_grid_private_method():
    """`sim._build_grid()` stays a bound method returning a Grid."""
    from rfx.api import Simulation

    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    grid = sim._build_grid()
    # Grid exposes a concrete shape — enough to prove the build ran.
    assert hasattr(grid, "shape")
    assert len(grid.shape) == 3
    assert all(n > 0 for n in grid.shape)


def test_assemble_materials_private_method():
    """`sim._assemble_materials(grid)` stays a bound method.

    Returns a tuple whose first element is the `MaterialArrays`. Empty
    geometry yields vacuum arrays — enough to prove the method is
    reachable and well-shaped. The exact tuple arity is intentionally
    NOT pinned here (it is an internal detail subject to change).
    """
    from rfx.api import Simulation

    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    grid = sim._build_grid()
    result = sim._assemble_materials(grid)
    assert isinstance(result, tuple)
    materials = result[0]
    assert hasattr(materials, "eps_r")
    assert materials.eps_r.shape == grid.shape


def test_build_nonuniform_grid_private_method():
    """`sim._build_nonuniform_grid()` stays a bound method.

    Constructed with a graded `dz_profile` (the only path that reaches
    `_build_nonuniform_grid`).
    """
    from rfx.api import Simulation

    dz = np.array([0.5e-3] * 5 + [0.4e-3] * 4, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3,
        dz_profile=dz,
        cpml_layers=4,
    )
    grid = sim._build_nonuniform_grid()
    assert hasattr(grid, "shape")
    assert len(grid.shape) == 3
    # nz includes CPML padding, so it is >= the supplied profile length.
    assert grid.nz >= len(dz)
