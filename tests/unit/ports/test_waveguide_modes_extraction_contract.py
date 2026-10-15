"""Contract: waveguide mode helpers live in rfx.model.waveguide_mode_solver.

The ten pure-NumPy helpers and the four dependencies imported by
rfx.eigenmode live in the model layer. Private helpers are not re-exported
from rfx.sources.waveguide_port.
"""

import pytest
import importlib

import numpy as np

# The 10 helpers extracted into rfx.model.waveguide_mode_solver.
_MOVED = (
    "_second_diff_1d",
    "_galerkin_stiffness_mass_1d",
    "_galerkin_eigh_separable_laplacian_2d",
    "_cell_centred_gradient",
    "_pick_eigenmode_by_overlap",
    "_pick_eigenmode_by_target_then_overlap",
    "_aperture_area",
    "_shift_profile_to_dual",
    "_orthonormalize_profile_arrays",
    "_scale_h_to_unit_cross",
)

# Symbols eigenmode.py imports from the model layer.
_EIGENMODE_PRIVATE_DEPS = (
    "_te_mode_profiles",
    "_tm_mode_profiles",
    "cutoff_frequency",
    "C0_LOCAL",
)


def test_moved_helpers_importable_from_new_module():
    mod = importlib.import_module("rfx.model.waveguide_mode_solver")
    missing = [n for n in _MOVED if not hasattr(mod, n)]
    assert not missing, f"_waveguide_modes is missing extracted symbols: {missing}"


def test_moved_helpers_reexported_with_same_identity():
    wp = importlib.import_module("rfx.sources.waveguide_port")
    for name in _MOVED:
        assert not hasattr(wp, name), f"waveguide_port re-exports private helper {name}"


def test_eigenmode_private_deps_stay_in_waveguide_port():
    modes = importlib.import_module("rfx.model.waveguide_modes")
    for name in _EIGENMODE_PRIVATE_DEPS:
        assert hasattr(modes, name), (
            f"rfx.model.waveguide_modes.{name} is imported by rfx/eigenmode.py "
            "and must be present"
        )
    # eigenmode.py must still import cleanly against that surface.
    importlib.import_module("rfx.eigenmode")


@pytest.mark.source_scan
def test_extracted_module_has_no_rfx_dependencies():
    """The extracted module is pure NumPy (no FDTD/AD coupling) — keep it that
    way so it can never reintroduce an import cycle through waveguide_port."""
    import rfx.model.waveguide_mode_solver as mod

    src = importlib.util.find_spec("rfx.model.waveguide_mode_solver").origin
    text = open(src).read()
    for forbidden in ("import jax", "from rfx", "jnp", "EPS_0", "MU_0"):
        assert forbidden not in text, (
            f"_waveguide_modes must stay pure-NumPy; found '{forbidden}'"
        )


def test_mode_solver_smoke():
    """A light functional smoke so the contract also exercises the code path."""
    from rfx.model.waveguide_mode_solver import _galerkin_eigh_separable_laplacian_2d, _galerkin_stiffness_mass_1d, _second_diff_1d

    uw = np.full(8, 1e-3)
    D = _second_diff_1d(uw, bc="dirichlet")
    assert D.shape == (8, 8)
    Ku, mu = _galerkin_stiffness_mass_1d(uw, bc="dirichlet")
    Kv, mv = _galerkin_stiffness_mass_1d(np.full(6, 1e-3), bc="dirichlet")
    evals, evecs = _galerkin_eigh_separable_laplacian_2d(Ku, mu, Kv, mv)
    assert evals.shape == (48,) and evecs.shape == (48, 48)
    # Dirichlet transverse Laplacian eigenvalues are non-negative.
    assert float(evals.min()) > -1e-6
