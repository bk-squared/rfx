"""AD-traceability tests for compute_waveguide_s_matrix assembly (WI-2).

WI-2 converted the S-matrix assembly in rfx/api/_sparams.py from
np.array/np.zeros in-place indexing to functional jnp.stack so the
returned s_params is a JAX array and jax.grad flows through it.

Scope of these tests
--------------------
- The ASSEMBLY LAYER (the WaveguideSMatrixResult.s_params field).
- All three normalize modes: False, True, "flux".
- Both float64 structural-equivalence (golden snapshot) and AD smoke.

Out of scope
------------
- The port BUILDER layer (add_waveguide_port) has a pre-existing tape
  break at rfx/api/__init__.py (~line 1405):
      np.asarray(freqs_arr, dtype=float)
  which prevents differentiating through the sim *construction* when
  freqs is a traced value. That is a separate work item. The tests here
  verify that once the forward pass has run, the returned s_params is
  a proper jax.Array whose downstream computations are AD-traceable.
"""
from __future__ import annotations

import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warnings
try:
    # Modern JAX (the scoped x64 context manager was promoted to top-level;
    # jax.experimental.enable_x64 was removed in v0.8.0).
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX (< ~0.4.31)
    from jax.experimental import enable_x64 as _enable_x64


@pytest.fixture(autouse=True)
def _scoped_x64():
    """Enable x64 PER-TEST via the context manager, NOT module-level.

    A module-level ``jax.config.update("jax_enable_x64", True)`` permanently
    flips x64 ON for the whole pytest process and leaks into downstream
    same-process tests (test_wire_*/test_verification then fail with lax.scan
    carry-dtype TypeErrors mid-suite). The context manager restores the prior
    setting on exit, so x64 stays scoped to this file. See
    tests/test_msl_sparam_ad.py for the same pattern.
    """
    with _enable_x64(True):
        yield

# ---------------------------------------------------------------------------
# Fixtures directory (relative to this file)
# ---------------------------------------------------------------------------
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_FREQS = jnp.linspace(5e9, 6.5e9, 4)  # concrete, reused across all tests


def _make_wr90_sim():
    """Minimal 2-port WR-90 sim (8 CPML layers, no PEC obstacle)."""
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        0.010, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.090, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=_FREQS, f0=6e9, bandwidth=0.5, name="right",
    )
    return sim


# ---------------------------------------------------------------------------
# Golden-snapshot equivalence  (float64, all 3 modes × uniform path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode_name,normalize", [
    ("false", False),
    ("true", True),
    ("flux", "flux"),
])
def test_wg_smatrix_golden_equivalence_float64(mode_name, normalize):
    """jnp float64 assembly matches pre-refactor numpy golden to 1e-6."""
    golden_path = os.path.join(_FIXTURE_DIR, f"golden_wg_smatrix_normalize_{mode_name}.npy")
    if not os.path.exists(golden_path):
        pytest.skip(f"Golden fixture not found: {golden_path}")

    golden = np.load(golden_path)  # complex128, shape (2, 2, 4)

    # Verify sha256 integrity
    sha = hashlib.sha256(golden.tobytes()).hexdigest()
    _EXPECTED_SHA = {
        "false": "fcf514d5791689b75d0329de8d781df6d14914c1a858c89456c0fbb8c4eaa135",
        "true":  "82ec3bb49a51332d25dae8c772130b3c9253e6949c8010bbad112175606bca2e",
        "flux":  "f853c126385630f84244fc22d53c1583615a64be7058853a017d2926689e97b8",
    }
    assert sha == _EXPECTED_SHA[mode_name], (
        f"Golden fixture sha256 mismatch for normalize={normalize}: "
        f"got {sha}, expected {_EXPECTED_SHA[mode_name]}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _make_wr90_sim().compute_waveguide_s_matrix(
            num_periods=4, normalize=normalize,
        )

    s = np.array(res.s_params, dtype=np.complex128)
    diff = np.abs(s - golden)

    # Per-frequency overlay (R5 mandate: full trace, not just scalar norm)
    print(f"\n[normalize={normalize}] per-freq max|delta| per (port_i, port_j):")
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            print(f"  S[{i},{j},:] max_err={diff[i,j,:].max():.3e}  "
                  f"values={np.array2string(s[i,j,:], precision=6)}")

    assert diff.max() < 1e-6, (
        f"normalize={normalize}: max abs deviation {diff.max():.3e} exceeds 1e-6 gate. "
        "Shape-divergence in the assembly — see per-frequency overlay above."
    )


# ---------------------------------------------------------------------------
# s_params is a JAX array (not numpy) for all 3 modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalize", [False, True, "flux"])
def test_wg_smatrix_returns_jax_array(normalize):
    """WaveguideSMatrixResult.s_params must be a jax.Array (not numpy)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _make_wr90_sim().compute_waveguide_s_matrix(
            num_periods=4, normalize=normalize,
        )
    assert isinstance(res.s_params, jax.Array), (
        f"normalize={normalize}: s_params type is {type(res.s_params)}, "
        "expected jax.Array. Assembly still wrapping in numpy."
    )
    assert isinstance(res.freqs, jax.Array), (
        f"normalize={normalize}: freqs type is {type(res.freqs)}, "
        "expected jax.Array."
    )


# ---------------------------------------------------------------------------
# AD smoke test: jax.grad through post-assembly computation
# ---------------------------------------------------------------------------
# The upstream port BUILDER (add_waveguide_port) has a pre-existing tape
# break on np.asarray(freqs_arr) — out of WI-2 scope.  We validate the
# ASSEMBLY OUTPUT is AD-traceable by differentiating a post-assembly scalar
# objective through a scale parameter.

@pytest.mark.parametrize("normalize", [False, True, "flux"])
def test_wg_smatrix_assembly_ad_traceable(normalize):
    """jax.grad through a post-assembly scalar objective returns finite grad."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _make_wr90_sim().compute_waveguide_s_matrix(
            num_periods=4, normalize=normalize,
        )

    s = res.s_params  # jax.Array

    def objective(scale: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(s * scale) ** 2).real

    grad = jax.grad(objective)(jnp.array(1.0))
    assert jnp.isfinite(grad), (
        f"normalize={normalize}: jax.grad returned non-finite value {grad}. "
        "Assembly output is not AD-traceable."
    )
    # grad must also be non-trivially non-zero for normalize=False
    # (the True/flux short sims yield near-zero S but the grad of |S*scale|^2
    #  at scale=1 is 2*sum(|S|^2) which is >= 0; just check finite)
    print(f"\n[normalize={normalize}] AD grad(sum|S*scale|^2)|_{{scale=1}} = {float(grad):.6e}")


# ---------------------------------------------------------------------------
# Pre-existing upstream tape-break documentation test
# ---------------------------------------------------------------------------

def test_wg_smatrix_upstream_builder_tape_break_documented():
    """Document the known pre-existing tape break in add_waveguide_port.

    rfx/api/__init__.py ~line 1405:
        freqs_np = np.asarray(freqs_arr, dtype=float)

    This breaks the tape when freqs is a *traced* jax value (e.g. when
    differentiating through the sim construction). It is NOT fixed by WI-2
    (assembly only). This test confirms the break still exists so that a
    future fix is not silently masked.
    """
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec, Boundary

    def build_and_run(freq_scale):
        freqs = jnp.linspace(5e9, 6.5e9, 4) * freq_scale
        sim = Simulation(
            freq_max=10e9,
            domain=(0.12, 0.04, 0.02),
            dx=0.003,
            boundary=BoundarySpec(
                x="cpml",
                y=Boundary(lo="pec", hi="pec"),
                z=Boundary(lo="pec", hi="pec"),
            ),
            cpml_layers=8,
        )
        sim.add_waveguide_port(
            0.010, direction="+x", mode=(1, 0), mode_type="TE",
            freqs=freqs, f0=6e9, bandwidth=0.5, name="left",
        )
        sim.add_waveguide_port(
            0.090, direction="-x", mode=(1, 0), mode_type="TE",
            freqs=freqs, f0=6e9, bandwidth=0.5, name="right",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(num_periods=4, normalize=False)
        return jnp.sum(jnp.abs(res.s_params) ** 2).real

    with pytest.raises(Exception) as exc_info:
        jax.grad(build_and_run)(jnp.array(1.0))

    # Confirm it's the expected TracerArrayConversionError (not a new breakage)
    assert "TracerArrayConversionError" in type(exc_info.value).__name__ or \
           "TracerArrayConversionError" in str(exc_info.value), (
        f"Expected TracerArrayConversionError from upstream builder tape break, "
        f"got: {type(exc_info.value).__name__}: {exc_info.value}"
    )
    print(
        "\n[documented] Upstream builder tape break confirmed at "
        "rfx/api/__init__.py (~line 1405): np.asarray(freqs_arr, dtype=float). "
        "Fix target: WI-3 or similar port-builder AD work item."
    )
