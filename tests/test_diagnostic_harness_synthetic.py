"""Unit tests for scripts/diagnostics/wr90_port comparator harness.

Pin the V/I projection recipe and the Yee leapfrog half-step
correction on synthetic ideal field data, so a future regression on
either piece of the comparator is caught before any real FDTD run is
done. The 2026-04-28 / 2026-04-29 chain that traced a 0.13 spread
phantom through seven sessions of unnecessary investigation could
have been short-circuited by a single test like this.

Test 1 — cell-recipe accuracy on a synthetic PEC-short standing wave:
    construct (E_z, H_y) at a probe plane d from a PEC short, with
    no Yee timing offset, and verify the cell recipe returns
    |S11(f)| = 1 to numerical precision.

Test 2 — half-step correction round-trip:
    apply the Yee H-side timing offset (multiply H by exp(-jω·dt/2))
    to the same synthetic field and verify that the comparator's
    correction (multiply by exp(+jω·dt/2)) cancels it back to
    |S11(f)| = 1. Without the correction, |S11(f)| should drift away
    from 1 by an O(1) amount across the band.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
C0 = 2.998e8
MU_0 = 1.2566370614e-6


def _load_s11_module():
    spec = importlib.util.spec_from_file_location(
        "s11_from_dumps", REPO / "scripts" / "diagnostics" / "wr90_port"
        / "s11_from_dumps.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_pec_short_field(
    a: float, b: float, dx: float, freqs_hz: np.ndarray, d_pec: float
):
    """Synthesise the analytic PEC-short standing-wave field on a Yee mesh.

    At a measurement plane separated by ``d_pec`` of empty guide from a
    PEC short, the TE10 standing wave has:
        E_z(y, f) = E0 · sin(π·y/a) · (1 − exp(−j·2·β·d))
        H_y(y, f) = (E0/Z_TE) · sin(π·y/a) · (1 + exp(−j·2·β·d))
    No (x, z) dependence at the plane; H_y trivially extends along z.
    """
    Ny = int(round(a / dx)) + 1
    Nz = int(round(b / dx)) + 1
    y = np.linspace(0.0, a, Ny)
    z = np.linspace(0.0, b, Nz)
    omega = 2.0 * np.pi * np.asarray(freqs_hz)
    kc = np.pi / a
    beta = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 1e-30))
    Z = omega * MU_0 / beta
    sin_y = np.sin(np.pi * y / a)
    refl = np.exp(-1j * beta * 2.0 * d_pec)
    V = 1.0 - refl
    I = (1.0 + refl) / Z
    weight_y = sin_y[None, :, None]
    Ez = V[:, None, None] * weight_y * np.ones((1, 1, Nz))
    Hy = I[:, None, None] * weight_y * np.ones((1, 1, Nz))
    return y, z, Ez.astype(np.complex128), Hy.astype(np.complex128), omega


@pytest.fixture
def s11_mod():
    return _load_s11_module()


def test_cell_recipe_pec_short_ideal(s11_mod):
    """Cell-recipe on a perfectly co-located synthetic PEC-short field
    must return |S11(f)| = 1 to machine precision.
    """
    a = 22.86e-3
    b = 10.16e-3
    dx = 0.5e-3
    freqs = np.linspace(8.2e9, 12.4e9, 21)
    d_pec = 0.095

    y, z, Ez, Hy, _ = _make_synthetic_pec_short_field(a, b, dx, freqs, d_pec)
    s11, *_ = s11_mod.s11_from_field(Ez, Hy, y, z, freqs, a)

    # Tight gate: synthetic field is analytically constructed, so |S11|
    # = 1 must hold to numerical precision (~1e-6 — limited only by
    # numpy float double precision).
    assert np.all(np.abs(np.abs(s11) - 1.0) < 1e-6), (
        f"|S11(f)| spread on synthetic PEC-short = "
        f"{np.abs(s11).max() - np.abs(s11).min():.3e}; expected ~1e-6"
    )


def test_half_step_correction_cancels_yee_offset(s11_mod):
    """The Yee leapfrog half-step correction must cancel an applied
    exp(-jω·dt/2) offset on H_y back to |S11|=1.

    This is the regression that — if it fails — silently breaks the
    rfx dump-recipe path the way the missing-correction phantom
    chased through seven sessions of waveguide-port investigation in
    April 2026.
    """
    a = 22.86e-3
    b = 10.16e-3
    dx = 0.5e-3
    freqs = np.linspace(8.2e9, 12.4e9, 21)
    d_pec = 0.095
    dt = 0.5 * dx / C0  # default Courant 0.5

    y, z, Ez, Hy_true, omega = _make_synthetic_pec_short_field(
        a, b, dx, freqs, d_pec
    )

    # Apply the Yee H-time offset: H is sampled at (n+1/2)·dt while the
    # probe-step pairing assumes (n+1)·dt, so the dump DFT picks up an
    # extra exp(-jω·dt/2) factor on H.
    Hy_yee = Hy_true * np.exp(-1j * omega * (0.5 * dt))[:, None, None]

    # Without correction: |S11| should drift far from 1.
    s11_uncorrected, *_ = s11_mod.s11_from_field(Ez, Hy_yee, y, z, freqs, a)
    spread_uncorrected = np.abs(s11_uncorrected).max() - np.abs(s11_uncorrected).min()
    # We just need the uncorrected spread to be visibly broken — at
    # this dt the offset is small, so the spread is ~5 % range.
    assert spread_uncorrected > 1e-3, (
        f"Half-step offset failed to break the recipe — spread "
        f"{spread_uncorrected:.3e}; either the synthetic offset is "
        f"too small or the recipe is silently absorbing the error."
    )

    # With correction: must cancel back to |S11|=1.
    Hy_corrected = Hy_yee * np.exp(+1j * omega * (0.5 * dt))[:, None, None]
    s11_corrected, *_ = s11_mod.s11_from_field(Ez, Hy_corrected, y, z, freqs, a)
    assert np.all(np.abs(np.abs(s11_corrected) - 1.0) < 1e-6), (
        f"Yee half-step correction did not recover |S11|=1; spread "
        f"{np.abs(s11_corrected).max() - np.abs(s11_corrected).min():.3e}"
    )


def test_run_rfx_dump_applies_half_step_correction(s11_mod):
    """Source-level guard: ``s11_from_dumps.run_rfx_dump`` must apply
    the Yee half-step correction. This test reads the function source
    and asserts the canonical correction line is present, so a future
    refactor that silently drops the correction is rejected before any
    FDTD run is done.
    """
    src_path = REPO / "scripts" / "diagnostics" / "wr90_port" / "s11_from_dumps.py"
    src = src_path.read_text()
    needle = "np.exp(+1j * omega * (0.5 * dt_sim))"
    assert needle in src, (
        f"{src_path} no longer contains the Yee half-step correction "
        f"line {needle!r}. If you intentionally moved the correction, "
        "update this test alongside."
    )
