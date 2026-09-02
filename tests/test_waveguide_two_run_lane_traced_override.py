"""Lane honesty for the two-run normalized waveguide lane (``normalize=True``).

The uniform ``normalize=True`` S-matrix is assembled on the host: the
extractor ``extract_waveguide_s_params_normalized`` in
``rfx/sources/waveguide_port.py`` converts the modal waves with
``np.array`` at three sites — the reference-run incident wave
(``np.array(a_inc_ref)``), the reference-run outgoing waves
(``np.array(b_ref_i)``) and the device-run outgoing waves
(``np.array(b_recv_dev)``). A design variable passed as a traced
``eps_override`` / ``sigma_override`` therefore cannot flow through this
lane. The v1.8 chain-closure plan (``docs/design_notes/v18_waveguide_s_chain_plan.md``,
WP1) asks for two things, in order:

1. a committed record of what the lane actually does under a tracer —
   which of the three sites raises, with the exception type and the
   traceback frame (this file, step 1);
2. a fail-fast ``NotImplementedError`` at the public dispatch in
   ``rfx/api/_sparams.py`` mirroring the non-uniform guard, so the
   extractor is never entered with a tracer (step 2 — this test flips to
   assert it).

Measured 2026-09-02 on main 378a9c95 (jax 0.6.2, CPU), tiny WR-90 2-port,
``n_steps=200``, ``jax.grad`` of ``sum|S(:, :, bin 2)|^2`` w.r.t. a scalar
scaling a whole-grid ``eps_override``:

    jax.errors.TracerArrayConversionError: The numpy.ndarray conversion
    method __array__() was called on traced array with shape complex64[4]
    frame: rfx/sources/waveguide_port.py in extract_waveguide_s_params_normalized
           b_recv_dev_np = np.array(b_recv_dev)

Only the DEVICE-run site fires. The two reference-run sites are reached
first and see concrete arrays: the vacuum reference run carries no design
variable, so its waves are never traced. Probe script (not committed):
the body of ``_grad_through_two_run_lane`` below is the probe.
"""

from __future__ import annotations

import traceback
import warnings

import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary

_EXTRACTOR = "extract_waveguide_s_params_normalized"
_DEVICE_RUN_SITE = "np.array(b_recv_dev)"
_REFERENCE_RUN_SITES = ("np.array(a_inc_ref)", "np.array(b_ref_i)")


def _wr90_sim():
    """Tiny 2-port WR-90 (40x13x7 cells), same fixture as the flux-lane AD test."""
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
    freqs = jnp.linspace(5e9, 6.5e9, 4)
    sim.add_waveguide_port(0.010, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=6e9, bandwidth=0.5, name="left")
    sim.add_waveguide_port(0.090, direction="-x", mode=(1, 0), mode_type="TE",
                           freqs=freqs, f0=6e9, bandwidth=0.5, name="right")
    return sim


def _grad_through_two_run_lane(override_kw: str):
    """``jax.grad`` through ``normalize=True`` with a traced override.

    Returns the exception the lane raised, or ``None`` if it ran.
    """
    sim = _wr90_sim()
    grid = sim._build_grid()
    base = jnp.ones(grid.shape, dtype=jnp.float32)
    if override_kw == "sigma_override":
        base = base * 1e-3  # a small, traced conductivity field

    def objective(alpha):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(
                n_steps=200, normalize=True, **{override_kw: base * alpha},
            )
        return jnp.real(jnp.sum(jnp.abs(res.s_params[:, :, 2]) ** 2))

    try:
        jax.grad(objective)(jnp.float32(1.0))
    except Exception as exc:  # noqa: BLE001 — the exception IS the measurement
        return exc
    return None


def _rfx_frames(exc: BaseException) -> list[traceback.FrameSummary]:
    """Traceback frames inside the two files the lane passes through."""
    return [
        f for f in traceback.extract_tb(exc.__traceback__)
        if f.filename.endswith(("waveguide_port.py", "_sparams.py"))
    ]


def _describe(exc: BaseException) -> str:
    lines = [f"{type(exc).__module__}.{type(exc).__name__}: {str(exc).splitlines()[0]}"]
    for f in _rfx_frames(exc):
        lines.append(f"  {f.filename.rsplit('/rfx/', 1)[-1]}:{f.lineno} in {f.name}: {f.line}")
    return "\n".join(lines)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "WP1 step 1 record (measured 2026-09-02, main 378a9c95): the lane raises "
        "jax.errors.TracerArrayConversionError ('The numpy.ndarray conversion "
        "method __array__() was called on traced array with shape complex64[4]') "
        "from extract_waveguide_s_params_normalized at the DEVICE-run site "
        "`b_recv_dev_np = np.array(b_recv_dev)`; the reference-run sites "
        "np.array(a_inc_ref) / np.array(b_ref_i) see concrete arrays. "
        "Flips green when the dispatch guard (WP1 step 2) raises "
        "NotImplementedError before the extractor is entered."
    ),
)
def test_two_run_lane_traced_eps_override_fails_fast_at_dispatch():
    exc = _grad_through_two_run_lane("eps_override")
    assert exc is not None, "the two-run lane ran under a tracer without raising"
    record = _describe(exc)
    print("\n[two-run lane, traced eps_override]\n" + record)

    if isinstance(exc, jax.errors.TracerArrayConversionError):
        # The measured raise site. A change here means the record above is
        # stale: pytest.fail is NOT an AssertionError, so the strict xfail
        # reports it as a real failure instead of swallowing it.
        frames = _rfx_frames(exc)
        hit = [f for f in frames if f.name == _EXTRACTOR and _DEVICE_RUN_SITE in (f.line or "")]
        if not hit:
            pytest.fail("raise site moved away from the device-run np.array:\n" + record)
        leaked = [f for f in frames if any(s in (f.line or "") for s in _REFERENCE_RUN_SITES)]
        if leaked:
            pytest.fail("a reference-run np.array site raised:\n" + record)

    # The contract (WP1 step 2): fail fast at the public entry, extractor
    # never entered with a tracer.
    assert isinstance(exc, NotImplementedError), (
        "expected NotImplementedError at compute_waveguide_s_matrix dispatch, got:\n" + record
    )
    assert not any(f.name == _EXTRACTOR for f in _rfx_frames(exc)), (
        "the extractor was entered with a tracer:\n" + record
    )
