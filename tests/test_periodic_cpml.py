"""Issue #68 — per-axis boundary selection: periodic x/y + CPML z.

Before the fix, `set_periodic_axes("xy")` combined with `boundary="cpml"`
allocated CPML psi arrays on all six faces. The CPML absorption on the
periodic faces fought the wrap-around, producing physically meaningless
results (S11 = 0 dB flat across the band) and a spurious preflight
warning about geometry "extending into the CPML region" on the periodic
axes.

After the fix, `_build_grid` computes the effective CPML axes by
removing the periodic axes, so `grid.cpml_axes == "z"` when
`periodic_axes == "xy"`. The preflight P1.9 geometry-in-CPML check
uses the per-axis CPML thickness so periodic axes (which now have
`cpml_thick_xyz[ax] == 0`) are naturally skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation


def _build_absorber_sim():
    """Normal-incidence absorber stand-in: periodic in xy, CPML in z."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, 0.02),
        dx=5e-4,
        cpml_layers=4,
        boundary="cpml",
    )
    sim.set_periodic_axes("xy")
    return sim


def test_build_grid_drops_periodic_axes_from_cpml():
    """Issue #68: `_build_grid` must not allocate CPML on periodic axes."""
    sim = _build_absorber_sim()
    grid = sim._build_grid()
    assert grid.cpml_axes == "z", (
        f"cpml_axes must exclude periodic 'xy', got {grid.cpml_axes!r}"
    )
    assert grid.pad_x == 0, f"pad_x must be 0 on periodic x, got {grid.pad_x}"
    assert grid.pad_y == 0, f"pad_y must be 0 on periodic y, got {grid.pad_y}"
    assert grid.pad_z == 4, (
        f"pad_z must preserve CPML layers on non-periodic z, got {grid.pad_z}"
    )


def test_build_grid_full_cpml_without_periodic_axes():
    """Regression: default CPML behaviour (no periodic) keeps all six faces."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, 0.02),
        dx=5e-4,
        cpml_layers=4,
        boundary="cpml",
    )
    grid = sim._build_grid()
    assert grid.cpml_axes == "xyz", (
        f"cpml_axes must be 'xyz' by default, got {grid.cpml_axes!r}"
    )
    assert (grid.pad_x, grid.pad_y, grid.pad_z) == (4, 4, 4)


def test_preflight_skips_periodic_axis_cpml_warning():
    """Issue #68: geometry spanning the whole periodic axis must not trip the
    "extends into CPML region" warning (there is no CPML on that axis).
    """
    from rfx import Box

    sim = _build_absorber_sim()
    sim.add_material("slab", eps_r=4.0)
    # Slab fills the whole periodic x/y extents but sits in the middle of z.
    sim.add(Box((0.0, 0.0, 0.008), (0.01, 0.01, 0.012)), material="slab")
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.015), "ez")

    issues = sim.preflight(strict=False)
    x_or_y_cpml_warnings = [
        msg for msg in issues
        if "extends into CPML region" in msg
        and (" x-axis" in msg or " y-axis" in msg)
    ]
    assert not x_or_y_cpml_warnings, (
        "periodic xy must not trigger CPML-region geometry warning on x or y; "
        f"got {x_or_y_cpml_warnings}"
    )


def test_preflight_still_warns_on_non_periodic_z_axis():
    """Regression: z-axis (non-periodic) CPML intrusion must still warn.

    Issue #500: CPML pads EXTERIOR to the requested domain, so a slab
    spanning z=[0, 0.0005] sits entirely INSIDE the physical domain
    (z=0.02) and can never touch the absorber — the original fixture here
    ("deep inside the z-CPML layer") encoded the same interior-frame
    assumption the #500 bug had, and used to false-fire. A genuine
    issue-#61 leak requires the box to actually cross the z=0 edge into
    the exterior absorber (which starts at z=0 and extends to z=-0.002
    given cpml_layers=4, dx=5e-4).
    """
    from rfx import Box

    sim = _build_absorber_sim()
    sim.add_material("slab", eps_r=4.0)
    # Slab straddles z=0: c1[2]=-0.0003 is genuinely in the exterior z_lo
    # absorber, c2[2]=0.0002 is interior.
    sim.add(Box((0.0, 0.0, -0.0003), (0.01, 0.01, 0.0002)), material="slab")

    issues = sim.preflight(strict=False)
    z_cpml_warnings = [
        msg for msg in issues
        if "extends into CPML region" in msg and " z-axis" in msg
    ]
    assert z_cpml_warnings, (
        "slab straddling z=0 should trigger z-axis CPML-region warning"
    )


def test_forward_periodic_xy_cpml_z_runs_without_nan():
    """End-to-end: periodic xy + CPML z forward must produce finite fields.

    Before the fix, the conflicting boundary conditions degenerated the
    simulation (documented as S11 = 0 dB flat on rfx-TAP ex6). The
    minimum smoke criterion here is that the probe time series is finite
    and non-zero after the fix.
    """
    sim = _build_absorber_sim()
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.010), "ez")
    fr = sim.forward(n_steps=80, skip_preflight=True)
    ts = np.asarray(fr.time_series)
    assert ts.shape[0] == 80
    assert np.all(np.isfinite(ts)), "periodic+CPML forward produced NaN/Inf"
    assert float(np.max(np.abs(ts))) > 0.0, (
        "source did not excite the probe — possible boundary conflict"
    )


@pytest.mark.parametrize("axes", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_build_grid_honors_arbitrary_periodic_axis_sets(axes):
    """`_build_grid` must complement `_periodic_axes` correctly for any subset."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.01, 0.01, 0.01),
        dx=5e-4,
        cpml_layers=4,
        boundary="cpml",
    )
    sim.set_periodic_axes(axes)
    grid = sim._build_grid()
    expected = "".join(ax for ax in "xyz" if ax not in axes)
    assert grid.cpml_axes == expected, (
        f"periodic={axes!r}: expected cpml_axes={expected!r}, got "
        f"{grid.cpml_axes!r}"
    )
