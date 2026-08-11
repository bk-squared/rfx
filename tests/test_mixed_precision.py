"""Tests for mixed precision (float16 fields, float32 accumulation) support.

Boundary coverage is deliberate (issue #644).  Until #644 this file was 100%
``boundary="pec"``, and PEC never constructs CPML state at all
(``rfx/simulation.py`` leaves ``apply_cpml_e``/``apply_cpml_h`` as ``None``),
so the entire CPML dtype path was untested — ``precision="mixed"`` + CPML had
in fact NEVER worked.  Every new dtype/absorption assertion below therefore
carries a CPML row, and ``test_precision_boundary_matrix`` pins the full 2x2
from the issue so the gap cannot silently reopen.

The file-level ``pytestmark = pytest.mark.gpu`` was also removed in #644.  It
was the SECOND reason the gap stayed invisible: ``pyproject.toml``'s
``addopts = "-m 'not gpu and not slow and not slow_physics'"`` deselected the
whole file from every default local run.  Nothing here needs a GPU — these are
dtype/dispatch assertions on 40**3-and-smaller grids, and the marker's own
definition is "tests that need GPU — too slow on CPU or GPU-specific".
Measured on CPU (``JAX_PLATFORMS=cpu``): the original 13 tests ran in 6.5 s
wall, slowest single test 1.27 s.
"""

import warnings

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import init_state, FDTDState, MaterialArrays, init_materials
from rfx.api import Simulation
from rfx.boundaries.cpml import init_cpml
from rfx.sources.sources import GaussianPulse


# ---------------------------------------------------------------------------
# Unit tests for core init_state
# ---------------------------------------------------------------------------

class TestInitState:
    def test_default_dtype_is_float32(self):
        state = init_state((5, 5, 5))
        assert state.ex.dtype == jnp.float32
        assert state.hx.dtype == jnp.float32

    def test_float16_fields(self):
        state = init_state((5, 5, 5), field_dtype=jnp.float16)
        assert state.ex.dtype == jnp.float16
        assert state.ey.dtype == jnp.float16
        assert state.ez.dtype == jnp.float16
        assert state.hx.dtype == jnp.float16
        assert state.hy.dtype == jnp.float16
        assert state.hz.dtype == jnp.float16
        # step counter stays int32 regardless
        assert state.step.dtype == jnp.int32

    def test_float16_memory_half_of_float32(self):
        shape = (50, 50, 50)
        s32 = init_state(shape, field_dtype=jnp.float32)
        s16 = init_state(shape, field_dtype=jnp.float16)
        bytes_32 = sum(f.nbytes for f in [s32.ex, s32.ey, s32.ez,
                                           s32.hx, s32.hy, s32.hz])
        bytes_16 = sum(f.nbytes for f in [s16.ex, s16.ey, s16.ez,
                                           s16.hx, s16.hy, s16.hz])
        assert bytes_16 == bytes_32 // 2


# ---------------------------------------------------------------------------
# Yee update tests with float16 state
# ---------------------------------------------------------------------------

class TestYeeUpdateMixedPrecision:
    def test_update_h_preserves_dtype(self):
        from rfx.core.yee import update_h
        state = init_state((10, 10, 10), field_dtype=jnp.float16)
        materials = init_materials((10, 10, 10))
        # Set a nonzero E field to get nonzero H update
        state = state._replace(ez=state.ez.at[5, 5, 5].set(jnp.float16(1.0)))
        new_state = update_h(state, materials, 1e-12, 1e-3)
        assert new_state.hx.dtype == jnp.float16
        assert new_state.hy.dtype == jnp.float16
        assert new_state.hz.dtype == jnp.float16

    def test_update_e_preserves_dtype(self):
        from rfx.core.yee import update_e
        state = init_state((10, 10, 10), field_dtype=jnp.float16)
        materials = init_materials((10, 10, 10))
        state = state._replace(hz=state.hz.at[5, 5, 5].set(jnp.float16(1.0)))
        new_state = update_e(state, materials, 1e-12, 1e-3)
        assert new_state.ex.dtype == jnp.float16
        assert new_state.ey.dtype == jnp.float16
        assert new_state.ez.dtype == jnp.float16

    def test_update_he_fast_preserves_dtype(self):
        from rfx.core.yee import update_he_fast, precompute_coeffs
        state = init_state((10, 10, 10), field_dtype=jnp.float16)
        materials = init_materials((10, 10, 10))
        coeffs = precompute_coeffs(materials, 1e-12, 1e-3, pec_axes="xyz")
        state = state._replace(ez=state.ez.at[5, 5, 5].set(jnp.float16(1.0)))
        new_state = update_he_fast(state, coeffs)
        assert new_state.ex.dtype == jnp.float16
        assert new_state.hx.dtype == jnp.float16


# ---------------------------------------------------------------------------
# API-level integration tests
# ---------------------------------------------------------------------------

class TestSimulationMixedPrecision:
    def test_precision_parameter_validation(self):
        with pytest.raises(ValueError, match="precision"):
            Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                       boundary="pec", precision="float128")

    def test_mixed_precision_runs(self):
        """Mixed precision simulation completes without NaN."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=100)
        assert result is not None
        # Fields should be float16
        assert result.state.ex.dtype == jnp.float16
        assert result.state.hx.dtype == jnp.float16
        # No NaN in final state
        assert not jnp.any(jnp.isnan(result.state.ez))
        assert not jnp.any(jnp.isnan(result.state.hz))

    def test_mixed_precision_accuracy(self):
        """Mixed precision should match float32 within reasonable tolerance.

        float16 has ~3.3 decimal digits of precision, so we compare
        the overall field pattern rather than requiring exact match.
        For a short simulation (50 steps), the relative error in the
        L2 norm of the fields should be small.
        """
        kwargs = dict(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
        )
        n_steps = 50

        sim32 = Simulation(**kwargs, precision="float32")
        sim32.add_source(position=(0.01, 0.01, 0.01), component="ez")
        r32 = sim32.run(n_steps=n_steps)

        sim16 = Simulation(**kwargs, precision="mixed")
        sim16.add_source(position=(0.01, 0.01, 0.01), component="ez")
        r16 = sim16.run(n_steps=n_steps)

        # Compare Ez field L2 norms
        ez32 = np.array(r32.state.ez, dtype=np.float32)
        ez16 = np.array(r16.state.ez, dtype=np.float32)

        norm32 = np.linalg.norm(ez32)
        if norm32 > 0:
            rel_error = np.linalg.norm(ez32 - ez16) / norm32
            # Allow up to 5% relative error for 50 steps with float16
            assert rel_error < 0.05, f"Relative error {rel_error:.4f} exceeds 5%"

    def test_mixed_precision_with_probe(self):
        """Probes should work in mixed precision mode."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=50)
        ts = np.array(result.time_series).ravel()
        assert len(ts) == 50
        # Probe should record nonzero values (source is active)
        assert np.any(ts != 0)
        assert not np.any(np.isnan(ts))

    def test_mixed_precision_memory_reduction(self):
        """Field arrays in mixed mode should be half the size of float32."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=10)
        st = result.state
        # Each field component should be float16 (2 bytes per element)
        assert st.ex.dtype == jnp.float16
        field_bytes = st.ex.nbytes
        expected_f32_bytes = st.ex.size * 4  # float32 would be 4 bytes
        assert field_bytes == expected_f32_bytes // 2

    def test_float32_precision_default(self):
        """Default precision='float32' should produce float32 fields."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=10)
        assert result.state.ex.dtype == jnp.float32

    def test_repr_includes_precision(self):
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        r = repr(sim)
        assert "precision='mixed'" in r


# ---------------------------------------------------------------------------
# Issue #644 — CPML dtype policy
#
# The psi_* arrays are ACCUMULATION state (a recursive convolution integrated
# over every timestep), so they are allocated at
# ``promote_types(field_dtype, float32)`` and never below float32.  Before the
# fix they followed ``field_dtype`` verbatim, which put them at float16 under
# ``precision="mixed"`` while the CPML coefficients stayed hard-pinned float32
# — ``psi = b*psi + c*curl`` then promoted float16 -> float32 and the lax.scan
# carry signature stopped matching its own input (TypeError).
#
# Note this is promote_types, NOT a flat float32 pin: psi followed field_dtype
# originally for a REASON (the #404 oblique Bloch path needs a complex carry),
# and a flat float32 pin would have fixed float16 by breaking complex64.
# ---------------------------------------------------------------------------

def _cpml_grid():
    return Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary="cpml",
    )._build_grid()


class TestCPMLPsiDtype:
    """Unit-level dtype policy — no FDTD stepping, runs in milliseconds."""

    @pytest.mark.parametrize("field_dtype,expected", [
        (jnp.float16, jnp.float32),     # issue #644: the fix
        (jnp.float32, jnp.float32),     # unchanged / bit-identical
        (jnp.complex64, jnp.complex64),  # issue #404 oblique Bloch — must survive
    ])
    def test_psi_dtype_is_promoted_to_at_least_float32(self, field_dtype, expected):
        _, st = init_cpml(_cpml_grid(), field_dtype=field_dtype)
        actual = {getattr(st, f).dtype for f in st._fields}
        assert actual == {jnp.dtype(expected)}, (
            f"field_dtype={jnp.dtype(field_dtype)} -> psi {actual}, "
            f"expected all {jnp.dtype(expected)}"
        )

    def test_psi_dtype_default_is_float32(self):
        _, st = init_cpml(_cpml_grid())
        assert {getattr(st, f).dtype for f in st._fields} == {jnp.dtype(jnp.float32)}

    def test_psi_never_drops_below_float32_for_any_real_field_dtype(self):
        """The invariant, stated directly: accumulation state is never float16."""
        for fd in (jnp.float16, jnp.float32):
            _, st = init_cpml(_cpml_grid(), field_dtype=fd)
            for f in st._fields:
                assert getattr(st, f).dtype != jnp.dtype(jnp.float16), (
                    f"{f} is float16 under field_dtype={jnp.dtype(fd)} — "
                    "a recursive accumulator must not sit in half precision"
                )


class TestMixedPrecisionCPML:
    """The coverage axis that was missing entirely before issue #644."""

    @staticmethod
    def _sim(precision, *, boundary="cpml", probe=False):
        sim = Simulation(
            freq_max=5e9, domain=(0.02, 0.02, 0.02),
            boundary=boundary, precision=precision,
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez",
                       amplitude_kind="field")
        if probe:
            sim.add_probe(position=(0.014, 0.01, 0.01), component="ez")
        return sim

    def test_mixed_precision_cpml_runs(self):
        """Issue #644 regression: this raised TypeError (lax.scan carry) before."""
        result = self._sim("mixed").run(n_steps=100)
        assert result is not None
        # The mode's promise is float16 STORAGE — the fix must not have bought
        # its way out of the crash by silently upcasting the fields.
        assert result.state.ex.dtype == jnp.float16
        assert result.state.hx.dtype == jnp.float16
        for f in ("ex", "ey", "ez", "hx", "hy", "hz"):
            assert not jnp.any(jnp.isnan(getattr(result.state, f)))
            assert not jnp.any(jnp.isinf(getattr(result.state, f)))

    def test_mixed_precision_cpml_forward_runs(self):
        """Entry-point parity: run() and forward() must both work (issue #644).

        This is also the tripwire for the ``forward()`` guard that PR #645
        added for this combination — if that guard is still present when this
        branch is rebased onto #645, this test goes red with
        ``NotImplementedError`` and forces the removal, which is exactly the
        intent.  Leaving the guard in would keep mixed+CPML blocked through
        ``forward()`` forever even though the root cause is fixed.
        """
        assert self._sim("mixed").forward(n_steps=10, skip_preflight=True) is not None

    def test_precision_boundary_matrix(self):
        """The full 2x2 from issue #644 — the cell that used to be TypeError."""
        seen = {}
        for boundary in ("pec", "cpml"):
            for precision in ("float32", "mixed"):
                st = self._sim(precision, boundary=boundary).run(n_steps=10).state
                seen[(boundary, precision)] = st.ex.dtype
        assert seen == {
            ("pec", "float32"): jnp.dtype(jnp.float32),
            ("pec", "mixed"): jnp.dtype(jnp.float16),
            ("cpml", "float32"): jnp.dtype(jnp.float32),
            ("cpml", "mixed"): jnp.dtype(jnp.float16),  # <- was TypeError
        }

    def test_mixed_precision_cpml_accuracy(self):
        """Agreement with float32 while the pulse is in the domain.

        Measured at the time of writing (40**3, dx=3.0 mm, dt=5.72 ps):
        0.20% at 50 steps, 0.23% at 100 steps.  The 5% gate matches the
        PEC sibling test above and sits ~25x above the measured value —
        it is a regression fence, not a fence drawn around the defect.
        """
        n_steps = 50
        ez32 = np.asarray(self._sim("float32").run(n_steps=n_steps).state.ez,
                          dtype=np.float64)
        ez16 = np.asarray(self._sim("mixed").run(n_steps=n_steps).state.ez,
                          dtype=np.float64)
        norm32 = np.linalg.norm(ez32)
        assert norm32 > 0, "float32 reference is identically zero — bad fixture"
        rel = np.linalg.norm(ez32 - ez16) / norm32
        assert rel < 0.05, f"mixed-vs-float32 relative L2 error {rel:.4%} exceeds 5%"

    def test_mixed_precision_cpml_still_absorbs(self):
        """float16 fields must not disable the absorber.

        This is the assertion that "it stops crashing" does not cover: a CPML
        that silently stopped absorbing would still run clean and still be
        float16.  Measured total-field-energy decay from peak (400 steps):
        float32 -76.3 dB, mixed -59.5 dB.  float16 IS materially worse — the
        quantization floor of the field storage sits ~17 dB above float32's
        residual — so the gate is set at -40 dB, which mixed clears by ~20 dB
        while a non-absorbing run (which stays near 0 dB) fails outright.
        """
        def energy(state):
            return float(sum(
                np.sum(np.asarray(getattr(state, f), dtype=np.float64) ** 2)
                for f in ("ex", "ey", "ez", "hx", "hy", "hz")
            ))

        peak = energy(self._sim("mixed").run(n_steps=100).state)
        tail = energy(self._sim("mixed").run(n_steps=400).state)
        assert peak > 0
        decay_db = 10.0 * np.log10(tail / peak)
        assert decay_db < -40.0, (
            f"mixed+CPML decayed only {decay_db:.1f} dB below peak — "
            "the absorber is not working in float16"
        )

    def test_mixed_precision_cpml_emits_no_unsafe_cast_warning(self):
        """The CPML corrections are float32 while the fields are float16.

        Scattering them straight in tripped JAX's "cannot safely cast value
        from dtype=float32 to dtype=float16" FutureWarning, which JAX states
        "will result in an error" in a future release.  apply_cpml_e/h now
        accumulate at the psi dtype and round back to storage dtype once.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._sim("mixed").run(n_steps=10)
        unsafe = [str(w.message) for w in caught
                  if "cannot safely cast" in str(w.message)]
        assert not unsafe, f"unsafe-cast warning(s) still emitted: {unsafe}"
