"""Issue #738 regression gate: preflight's waveguide-port checks must read
the RASTERIZED aperture the solve uses, not the declared geometry.

Root cause (#738, family #737): ``_check_waveguide_port_evanescent`` derived
its ``a``/``b`` from ``entry.*_range`` / ``self._domain`` -- both DECLARED
numbers. On a grid whose ``dx`` does not divide the declared width, three
different widths coexist and preflight saw none of the two that matter:

  examples/inverse_design/differentiable_s11_design.py, dx = 2 mm
    declared _WR90_A     22.860 mm   (what preflight checked)
    port aperture        22.000 mm   (WaveguidePort.a -> mode template, f_cutoff)
    rasterized guide     24.000 mm   (grid interior -> which modes exist)
  -> 11.5 GHz vs 0.90 x fc_next: 11.803 GHz on the declared guide (silent)
                                 11.242 GHz on the rasterized guide (violated)

This module is the enumerate-and-classify gate: it enumerates the port
surface dynamically and fails on any (declared, aperture, guide) triple
that is not in the classification table below.
"""
from __future__ import annotations

import contextlib
import io

import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

_A_WR90 = 22.86e-3
_B_WR90 = 10.16e-3

# cpml_layers is a budget SHARED across all six faces (issue #647) even
# though only x is an absorbing axis here (y/z are PEC) -- a value sized
# for the x extent can still exceed the y/z axes' own cell counts and
# fire an unrelated `absorber_budget_exceeds_axis` finding. Kept small
# enough (4) to clear the smallest y/z axis this file builds (WR-90 b at
# dx=2mm -> 7 interior cells) so every fixture here is preflight-clean on
# everything except the #738 surface under test.
_CPML_LAYERS = 4


def _pec_walls():
    return BoundarySpec(x="cpml",
                        y=Boundary(lo="pec", hi="pec"),
                        z=Boundary(lo="pec", hi="pec"))


def _build(*, dx, domain, y_range=None, z_range=None, freqs, f0=None):
    sim = Simulation(freq_max=12e9, domain=domain, dx=dx,
                      boundary=_pec_walls(), cpml_layers=_CPML_LAYERS)
    kw = {}
    if y_range is not None:
        kw["y_range"] = y_range
    if z_range is not None:
        kw["z_range"] = z_range
    sim.add_waveguide_port(0.024, direction="+x", mode=(1, 0), mode_type="TE",
                            freqs=freqs, f0=f0 or 9.75e9, name="p0", **kw)
    return sim


def _issues(sim):
    with contextlib.redirect_stdout(io.StringIO()):
        return list(sim.preflight())


def _codes(sim):
    return [getattr(i, "code", None) for i in _issues(sim)]


# --------------------------------------------------------------------------
# 1. The named defect: a port whose declared width dx does not divide must
#    raise a finding naming BOTH numbers.
# --------------------------------------------------------------------------

def test_non_dividing_declared_width_names_both_numbers():
    sim = _build(dx=2e-3, domain=(0.10, _A_WR90, _B_WR90),
                 y_range=(0.0, _A_WR90), z_range=(0.0, _B_WR90),
                 freqs=jnp.linspace(8e9, 11.5e9, 8))
    issues = _issues(sim)
    snaps = [i for i in issues if getattr(i, "code", None) == "port_aperture_snap"]
    assert snaps, (
        f"declared a={_A_WR90*1e3:.2f} mm on dx=2 mm rasterizes to a 22.000 mm "
        f"aperture inside a 24.000 mm guide; preflight must say so. "
        f"issues={[str(i) for i in issues]!r}"
    )
    text = " ".join(str(i) for i in snaps)
    assert "22.8600" in text and "22.0000" in text and "24.0000" in text, (
        f"the finding must name the DECLARED width and the rasterized "
        f"aperture/guide it is compared against; got {text!r}"
    )


def test_margin_heuristic_is_evaluated_on_the_rasterized_guide():
    """11.5 GHz clears 0.90 x fc_next on the declared 22.86 mm guide and
    violates it on the 24.000 mm guide the solve actually rasterizes."""
    sim = _build(dx=2e-3, domain=(0.10, _A_WR90, _B_WR90),
                 y_range=(0.0, _A_WR90), z_range=(0.0, _B_WR90),
                 freqs=jnp.linspace(8e9, 11.5e9, 8))
    ev = [i for i in _issues(sim) if getattr(i, "code", None) == "port_evanescent"]
    assert ev, "0.90 x fc_next on the rasterized guide is violated and must warn"
    assert "11.242" in str(ev[0]), (
        f"threshold must come from the rasterized guide (11.242 GHz), "
        f"not the declared one (11.803 GHz); got {str(ev[0])!r}"
    )


def test_dividing_declared_width_stays_silent():
    """dx divides the declared width exactly -> all three widths agree ->
    no snap finding. Guards against a checker that always fires."""
    sim = _build(dx=1e-3, domain=(0.10, 0.020, 0.010),
                 y_range=(0.0, 0.020), z_range=(0.0, 0.010),
                 freqs=jnp.asarray([8e9]), f0=8e9)
    assert "port_aperture_snap" not in _codes(sim)


# --------------------------------------------------------------------------
# 2. Enumerate-and-classify: every (declared, aperture, guide) triple the
#    port surface can produce must land in the table.
# --------------------------------------------------------------------------

_CASES = [
    # (id, dx, domain, y_range, z_range)
    ("wr90_dx2_explicit", 2e-3, (0.10, _A_WR90, _B_WR90),
     (0.0, _A_WR90), (0.0, _B_WR90)),
    ("wr90_dx2_default", 2e-3, (0.10, _A_WR90, _B_WR90), None, None),
    ("wr90_dx1_explicit", 1e-3, (0.10, _A_WR90, _B_WR90),
     (0.0, _A_WR90), (0.0, _B_WR90)),
    ("exact_dx1_explicit", 1e-3, (0.10, 0.020, 0.010),
     (0.0, 0.020), (0.0, 0.010)),
    ("exact_dx1_default", 1e-3, (0.10, 0.020, 0.010), None, None),
    ("exact_dx2_explicit", 2e-3, (0.10, 0.020, 0.010),
     (0.0, 0.020), (0.0, 0.010)),
    ("battery_dx3_default", 3e-3, (0.10, 0.040, 0.020), None, None),
    # Round-up case (reviewer-required): _range_to_slice's explicit
    # branch ROUNDS range endpoints to the nearest cell, so an aperture
    # can snap ABOVE the declared width too, not only below it (as every
    # other APERTURE_SNAP case above does). domain y is intentionally
    # larger than the port's declared y_range so the range's upper edge
    # rounds up into the extra room: 0.02286/0.001 = 22.86 -> rounds to
    # the 23rd cell -> aperture 23.000 mm > declared 22.860 mm.
    # port_aperture_snap is DIRECTION-AGNOSTIC by construction --
    # `_check_waveguide_port_aperture_snap` fires on
    # ``declared != aperture``, never on the sign of the difference --
    # so this is exercised as an additional APERTURE_SNAP case rather
    # than a new table row.
    ("wr90_roundup_dx1_y_only", 1e-3, (0.10, 0.030, 0.010),
     (0.0, _A_WR90), None),
]

_TABLE = {
    # classification -> whether a port_aperture_snap finding is required
    "EXACT": False,          # declared == aperture == guide
    "APERTURE_SNAP": True,   # declared != aperture (explicit range, dx-snap;
                             # either direction -- see wr90_roundup above)
    "GUIDE_SNAP": True,      # declared == aperture != guide (#729 site 2)
    "UNRASTERIZABLE": True,  # _range_to_slice rejects the range outright
}


def _classify(declared, aperture, guide, tol=1e-12):
    if aperture is None:
        return "UNRASTERIZABLE"
    if abs(declared - aperture) <= tol and abs(declared - guide) <= tol:
        return "EXACT"
    if abs(declared - aperture) > tol:
        return "APERTURE_SNAP"
    return "GUIDE_SNAP"


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_every_port_triple_is_classified(case):
    _id, dx, domain, y_range, z_range = case
    sim = _build(dx=dx, domain=domain, y_range=y_range, z_range=z_range,
                 freqs=jnp.asarray([9e9]), f0=9e9)
    grid = sim._build_grid()
    entry = sim._waveguide_ports[0]
    # Computed from COMMITTED primitives only (grid + _range_to_slice), so
    # this gate fails on an assertion pre-fix rather than on a missing
    # helper -- the failure it must catch is preflight staying silent.
    spans = {}
    for ax in "yz":
        ai = "xyz".index(ax)
        rng = getattr(entry, f"{ax}_range")
        n_axis = (grid.nx, grid.ny, grid.nz)[ai]
        pad_lo, pad_hi = grid.face_pads[2 * ai], grid.face_pads[2 * ai + 1]
        guide = float((n_axis - pad_lo - pad_hi - 1) * grid.dx)
        declared = float(rng[1] - rng[0]) if rng is not None else float(sim._domain[ai])
        try:
            _, aperture = sim._range_to_slice(
                rng, sim._domain[ai], grid.dx, n_axis, grid.axis_pads[ai])
        except ValueError:
            aperture = None
        spans[ax] = (declared, aperture if aperture is None else float(aperture), guide)
    codes = _codes(sim)
    for axis, (declared, aperture, guide) in sorted(spans.items()):
        kind = _classify(declared, aperture, guide)
        assert kind in _TABLE, (
            f"{_id}/{axis}: unclassified span triple "
            f"declared={declared} aperture={aperture} guide={guide} -- extend "
            f"the classification table or the surface grew a new shape"
        )
        if _TABLE[kind]:
            expect = ("port_aperture_unrasterizable"
                      if kind == "UNRASTERIZABLE" else "port_aperture_snap")
            assert expect in codes, (
                f"{_id}/{axis} classified {kind} "
                f"(declared={declared*1e3:.4f} aperture={aperture} "
                f"guide={(guide*1e3):.4f} mm) but preflight did not emit "
                f"{expect}; "
                f"codes={codes!r}"
            )
    if all(_classify(*spans[ax]) == "EXACT" for ax in spans):
        assert "port_aperture_snap" not in codes, (
            f"{_id}: every axis EXACT but preflight fired a snap finding; "
            f"codes={codes!r}"
        )
