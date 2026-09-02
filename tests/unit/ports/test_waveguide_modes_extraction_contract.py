"""Contract: the waveguide mode-solver extraction preserves every import path.

The pure-NumPy transverse mode solver + mode-profile algebra were extracted
verbatim from ``rfx/sources/waveguide_port.py`` into the sibling module
``rfx/sources/_waveguide_modes.py`` (off the FDTD/JAX AD tape). The extraction
re-imports every moved name back into the ``waveguide_port`` namespace so that
existing import paths keep resolving. This test pins that surface so a future
edit that drops or renames a re-export fails fast (CI), and confirms the two
sibling modules expose the SAME object (a real re-export, not a divergent copy).

It also pins that the symbols ``rfx/eigenmode.py`` imports privately from
``waveguide_port`` (``_te_mode_profiles``, ``_tm_mode_profiles``,
``cutoff_frequency``, ``C0_LOCAL``) were NOT moved — they must stay importable
from ``waveguide_port`` or eigenmode.py breaks.
"""

import importlib

import numpy as np

# The 10 helpers extracted into rfx.sources._waveguide_modes.
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

# Symbols eigenmode.py imports privately from waveguide_port — must NOT have moved.
_EIGENMODE_PRIVATE_DEPS = (
    "_te_mode_profiles",
    "_tm_mode_profiles",
    "cutoff_frequency",
    "C0_LOCAL",
)


def test_moved_helpers_importable_from_new_module():
    mod = importlib.import_module("rfx.sources._waveguide_modes")
    missing = [n for n in _MOVED if not hasattr(mod, n)]
    assert not missing, f"_waveguide_modes is missing extracted symbols: {missing}"


def test_moved_helpers_reexported_with_same_identity():
    wp = importlib.import_module("rfx.sources.waveguide_port")
    modes = importlib.import_module("rfx.sources._waveguide_modes")
    for name in _MOVED:
        assert hasattr(wp, name), f"waveguide_port no longer re-exports {name}"
        assert getattr(wp, name) is getattr(modes, name), (
            f"{name} re-export diverged from _waveguide_modes (must be the same object)"
        )


def test_eigenmode_private_deps_stay_in_waveguide_port():
    wp = importlib.import_module("rfx.sources.waveguide_port")
    for name in _EIGENMODE_PRIVATE_DEPS:
        assert hasattr(wp, name), (
            f"waveguide_port.{name} is imported privately by rfx/eigenmode.py "
            "and must not be moved/dropped"
        )
    # eigenmode.py must still import cleanly against that surface.
    importlib.import_module("rfx.eigenmode")


def test_extracted_module_has_no_rfx_dependencies():
    """The extracted module is pure NumPy (no FDTD/AD coupling) — keep it that
    way so it can never reintroduce an import cycle through waveguide_port."""
    import rfx.sources._waveguide_modes as mod

    src = importlib.util.find_spec("rfx.sources._waveguide_modes").origin
    text = open(src).read()
    for forbidden in ("import jax", "from rfx", "jnp", "EPS_0", "MU_0"):
        assert forbidden not in text, (
            f"_waveguide_modes must stay pure-NumPy; found '{forbidden}'"
        )


def test_mode_solver_smoke():
    """A light functional smoke so the contract also exercises the code path."""
    from rfx.sources._waveguide_modes import (
        _galerkin_eigh_separable_laplacian_2d,
        _galerkin_stiffness_mass_1d,
        _second_diff_1d,
    )

    uw = np.full(8, 1e-3)
    D = _second_diff_1d(uw, bc="dirichlet")
    assert D.shape == (8, 8)
    Ku, mu = _galerkin_stiffness_mass_1d(uw, bc="dirichlet")
    Kv, mv = _galerkin_stiffness_mass_1d(np.full(6, 1e-3), bc="dirichlet")
    evals, evecs = _galerkin_eigh_separable_laplacian_2d(Ku, mu, Kv, mv)
    assert evals.shape == (48,) and evecs.shape == (48, 48)
    # Dirichlet transverse Laplacian eigenvalues are non-negative.
    assert float(evals.min()) > -1e-6
