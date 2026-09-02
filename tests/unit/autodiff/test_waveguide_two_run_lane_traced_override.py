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
WP1) asked for two things, in order:

1. a committed record of what the lane actually did under a tracer —
   which of the three sites raised, with the exception type and the
   traceback frame (step 1, the record below);
2. a fail-fast ``NotImplementedError`` at the public dispatch in
   ``rfx/api/_sparams.py`` mirroring the non-uniform guard, so the
   extractor is never entered with a tracer (step 2 — what this file now
   asserts).

Step-1 record, measured 2026-09-02 on main 378a9c95 (jax 0.6.2, CPU),
tiny WR-90 2-port, ``n_steps=200``, ``jax.grad`` of ``sum|S(:, :, bin 2)|^2``
w.r.t. a scalar scaling a whole-grid ``eps_override``, before the guard:

    jax.errors.TracerArrayConversionError: The numpy.ndarray conversion
    method __array__() was called on traced array with shape complex64[4]
    frame: rfx/sources/waveguide_port.py in extract_waveguide_s_params_normalized
           b_recv_dev_np = np.array(b_recv_dev)

Only the DEVICE-run site fired. The two reference-run sites were reached
first and saw concrete arrays: the vacuum reference run carries no design
variable, so its waves are never traced.

The guard is scoped to TRACED overrides. A concrete ``eps_override`` on
this lane still runs forward (third test), so the guard cannot widen
silently into a forward-mode rejection.
"""

from __future__ import annotations

import traceback
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary

_EXTRACTOR = "extract_waveguide_s_params_normalized"
_DISPATCH = "compute_waveguide_s_matrix"


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


def _override_base(sim, override_kw: str):
    base = jnp.ones(sim._build_grid().shape, dtype=jnp.float32)
    if override_kw == "sigma_override":
        base = base * 1e-3  # a small conductivity field
    return base


def _grad_through_two_run_lane(override_kw: str):
    """``jax.grad`` through ``normalize=True`` with a traced override.

    Returns the exception the lane raised, or ``None`` if it ran. This is
    the step-1 probe, unchanged.
    """
    sim = _wr90_sim()
    base = _override_base(sim, override_kw)

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


@pytest.mark.parametrize("override_kw", ["eps_override", "sigma_override"])
def test_two_run_lane_traced_override_fails_fast_at_dispatch(override_kw):
    """A traced override on normalize=True raises NotImplementedError at the
    public dispatch, naming the lane and the override; the extractor is
    never entered (step-1 record: it used to raise a
    TracerArrayConversionError from the device-run np.array site)."""
    exc = _grad_through_two_run_lane(override_kw)
    assert exc is not None, "the two-run lane ran under a tracer without raising"
    record = _describe(exc)
    print(f"\n[two-run lane, traced {override_kw}]\n" + record)

    assert isinstance(exc, NotImplementedError), (
        "expected NotImplementedError at compute_waveguide_s_matrix dispatch, got:\n" + record
    )
    msg = str(exc)
    assert "normalize=True" in msg and override_kw in msg, msg
    assert "normalize='flux'" in msg and "normalize=False" in msg, msg
    frames = _rfx_frames(exc)
    assert frames and frames[-1].name == _DISPATCH, record
    assert not any(f.name == _EXTRACTOR for f in frames), (
        "the extractor was entered with a tracer:\n" + record
    )


def test_two_run_lane_concrete_eps_override_still_runs_forward():
    """The guard is tracer-scoped: a concrete eps_override on normalize=True
    still produces a finite S-matrix (forward use of the override channel)."""
    sim = _wr90_sim()
    eps = _override_base(sim, "eps_override") * 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(
            n_steps=200, normalize=True, eps_override=eps,
        )
    s = np.asarray(res.s_params)
    assert s.shape == (2, 2, 4), s.shape
    assert np.all(np.isfinite(s)), s
    print(f"\n[two-run lane, concrete eps_override] |S| per bin:\n{np.abs(s)}")
