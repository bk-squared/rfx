"""Issue #738 regression gate: preflight's waveguide-port checks must read
the RASTERIZED geometry the solve uses, not the declared one.

Root cause (#738, family #737): ``_check_waveguide_port_evanescent`` derived
its ``a``/``b`` from ``entry.*_range`` / ``self._domain`` -- both DECLARED
numbers. On a grid whose ``dx`` does not divide the declared width the
declared number and the number the solve uses are different:

  examples/inverse_design/differentiable_s11_design.py, at its
  then-committed dx = 2 mm (that example now carries dx = 1.27 mm, which
  divides both WR-90 walls exactly)
    declared _WR90_A     22.860 mm   (what preflight checked)
    port slice covers    22.000 mm   (WaveguidePort.a -> mode template, f_cutoff)
  -> preflight printed "All checks passed"

Two numbers are compared, and they answer different questions:

  declared    what the config states;
  rasterized  the span the port's grid slice actually covers,
              ``(hi_idx - lo_idx - 1) * dx``. ``port_aperture_snap`` fires
              on ``declared != rasterized`` and on nothing else.

A THIRD number, the guide, decides which higher-order modes exist and so
sets the 0.90 x fc_next margin heuristic. It is measured wall-to-wall on
the assembled ``pec_mask`` along the port's own transverse line
(``guide_source="pec_walls"``), or from the domain when the axis' two
faces are both PEC/PMC (``"domain_faces"``), or -- when neither holds --
it falls back to the port's own rasterized aperture (``"aperture"``).
The first version of this fix used the transverse DOMAIN extent
unconditionally, which is wrong for every sub-aperture port; the
_tj_device-shaped case below is the measured counter-example that pins
it.
"""
from __future__ import annotations

import contextlib
import io

import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

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


def _sub_aperture_sim(freqs, f0=5.5e9):
    """Sub-aperture port: the guide walls are interior PEC Boxes, not the
    domain faces. Geometry copied from
    tests/unit/sparams/test_waveguide_port_reference_sims.py::_tj_device's straight
    horizontal leg -- the committed shape the domain-extent guide got
    wrong (it reported fc_TE10 = 1.249 GHz for the 120 mm DOMAIN instead
    of the guide the PEC Boxes actually leave)."""
    sim = Simulation(freq_max=10e9, domain=(0.12, 0.12, 0.02),
                     boundary="cpml", cpml_layers=10, dx=0.002)
    sim.add(Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)), material="pec")
    sim.add(Box((0.0, 0.08, 0.0), (0.12, 0.12, 0.02)), material="pec")
    sim.add_waveguide_port(
        0.01, direction="+x", mode=(1, 0), mode_type="TE",
        y_range=(0.04, 0.08), z_range=(0.0, 0.02),
        freqs=freqs, f0=f0, ref_offset=3, probe_offset=15, name="left",
    )
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
        f"aperture; preflight must say so. "
        f"issues={[str(i) for i in issues]!r}"
    )
    text = " ".join(str(i) for i in snaps)
    assert "22.8600" in text and "22.0000" in text, (
        f"the finding must name the DECLARED width and the rasterized span "
        f"it is compared against; got {text!r}"
    )


def test_margin_heuristic_is_evaluated_on_the_rasterized_guide():
    """11.5 GHz clears 0.90 x fc_next on the declared 22.86 mm guide and
    violates it on the 24.000 mm guide this PEC-walled domain rasterizes."""
    sim = _build(dx=2e-3, domain=(0.10, _A_WR90, _B_WR90),
                 y_range=(0.0, _A_WR90), z_range=(0.0, _B_WR90),
                 freqs=jnp.linspace(8e9, 11.5e9, 8))
    ev = [i for i in _issues(sim) if getattr(i, "code", None) == "port_evanescent"]
    assert ev, "0.90 x fc_next on the rasterized guide is violated and must warn"
    assert "11.242" in str(ev[0]), (
        f"threshold must come from the rasterized guide (11.242 GHz), "
        f"not the declared one (11.803 GHz); got {str(ev[0])!r}"
    )
    assert "domain_faces" in str(ev[0]), (
        f"this axis IS closed by its two PEC domain faces, so the finding "
        f"must say where the guide came from; got {str(ev[0])!r}"
    )


def test_dividing_declared_width_stays_silent():
    """dx divides the declared width exactly -> declared == rasterized ->
    no snap finding. Guards against a checker that always fires."""
    sim = _build(dx=1e-3, domain=(0.10, 0.020, 0.010),
                 y_range=(0.0, 0.020), z_range=(0.0, 0.010),
                 freqs=jnp.asarray([8e9]), f0=8e9)
    assert "port_aperture_snap" not in _codes(sim)


# --------------------------------------------------------------------------
# 2. Sub-aperture ports: the guide is the walls, not the domain.
#    (Issue #738 review, blocking items 1-3. Each of the three asserts
#    below was verified to fail against the domain-extent version of
#    _port_transverse_spans.)
# --------------------------------------------------------------------------

def test_sub_aperture_port_does_not_fire_a_snap_finding():
    """declared == rasterized on both transverse axes -> nothing snapped,
    so port_aperture_snap must stay silent even though the port aperture
    is much narrower than the domain."""
    sim = _sub_aperture_sim(jnp.linspace(4.5e9, 6.5e9, 3))
    codes = _codes(sim)
    assert "port_aperture_snap" not in codes, (
        f"40 mm declared / 2 mm cells / 40 mm rasterized: nothing snapped. "
        f"codes={codes!r}"
    )
    # The '#729 site 2' note describes the ``value_range is None`` branch.
    # This port declares y_range AND z_range; the first version of this
    # fix keyed the note on ``declared == aperture`` and printed it here.
    texts = [str(i) for i in _issues(sim)]
    assert not any("no explicit range" in t for t in texts), texts


def test_sub_aperture_guide_is_measured_from_the_pec_walls():
    """The interior PEC Boxes leave a guide ~40 mm wide inside a 120 mm
    domain. The cutoffs the finding quotes must come from the walls."""
    sim = _sub_aperture_sim(jnp.linspace(4.5e9, 6.5e9, 3))
    ev = [i for i in _issues(sim)
          if getattr(i, "code", None) == "port_evanescent"]
    assert ev, "6.5 GHz exceeds 0.90 x fc_TE20 on the walled guide"
    text = str(ev[0])
    assert "pec_walls" in text, (
        f"the guide must be measured on the assembled pec_mask along the "
        f"port's transverse line; got {text!r}"
    )
    # Wall-to-wall on the mask: the PEC Boxes rasterize to y-nodes 10..29
    # and 50..69, so the electric walls the SOLVE sees sit at nodes 29 and
    # 50 -> (50 - 29) * 2 mm = 42.0 mm. fc_TE20 = c / 42.0 mm = 7.138 GHz,
    # threshold 6.424 GHz. The domain-extent version reported 120 mm ->
    # fc_TE20 = 2.498 GHz, threshold 2.248 GHz.
    assert "42.0000" in text and "7.138" in text and "6.424" in text, (
        f"expected the wall-measured 42.0000 mm guide (fc_TE20 7.138 GHz, "
        f"threshold 6.424 GHz); got {text!r}"
    )
    assert "2.248" not in text and "1.249" not in text, (
        f"the 120 mm DOMAIN must not be used as the guide; got {text!r}"
    )


def test_explicit_range_is_not_labelled_a_none_range():
    """The '#729 site 2' note describes the ``value_range is None`` branch
    of _range_to_slice. It must be keyed on that branch, not inferred from
    a width comparison."""
    explicit = _build(dx=2e-3, domain=(0.10, _A_WR90, _B_WR90),
                      y_range=(0.0, _A_WR90), z_range=(0.0, _B_WR90),
                      freqs=jnp.asarray([9e9]), f0=9e9)
    snaps = [str(i) for i in _issues(explicit)
             if getattr(i, "code", None) == "port_aperture_snap"]
    assert snaps
    assert not any("no explicit range" in s for s in snaps), (
        f"this port declares y_range/z_range explicitly; got {snaps!r}"
    )

    default = _build(dx=2e-3, domain=(0.10, _A_WR90, _B_WR90),
                     freqs=jnp.asarray([9e9]), f0=9e9)
    snaps = [str(i) for i in _issues(default)
             if getattr(i, "code", None) == "port_aperture_snap"]
    assert snaps and all("no explicit range" in s for s in snaps), (
        f"this port leaves both transverse ranges unset -- the note belongs "
        f"here; got {snaps!r}"
    )


def test_aperture_can_snap_above_the_declared_width():
    """_range_to_slice's explicit branch ROUNDS to the nearest node, so the
    rasterized span lands above the declared width as readily as below.
    This is what makes the #150 lower bounds' move onto the aperture a
    two-directional change rather than a relaxation."""
    sim = _build(dx=1e-3, domain=(0.10, 0.030, 0.010),
                 y_range=(0.0, _A_WR90), freqs=jnp.asarray([9e9]), f0=9e9)
    grid = sim._build_grid()
    entry = sim._waveguide_ports[0]
    slc, _ = sim._range_to_slice(entry.y_range, sim._domain[1], grid.dx,
                                 grid.ny, grid.axis_pads[1])
    rasterized = (slc[1] - slc[0] - 1) * grid.dx
    assert rasterized > _A_WR90, (
        f"expected the 22.860 mm range to round UP to 23.000 mm at "
        f"dx=1 mm; got {rasterized * 1e3:.4f} mm"
    )
    snaps = [str(i) for i in _issues(sim)
             if getattr(i, "code", None) == "port_aperture_snap"]
    assert any("22.8600" in t and "23.0000" in t for t in snaps), snaps


# --------------------------------------------------------------------------
# 3. Enumerate-and-classify: every (declared, rasterized) pair the port
#    surface can produce must land in the table.
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
    # Sub-aperture (reviewer-required): an explicit range NARROWER than
    # the domain. This is a committed pattern
    # (tests/unit/sparams/test_waveguide_port_reference_sims.py, tests/unit/api/test_api.py,
    # tests/unit/runners/test_distributed.py) and it was the table's blind spot -- the
    # first version of this fix fired a snap finding on all three of that
    # file's ports with declared == aperture.
    ("sub_aperture_dx2", 2e-3, (0.10, 0.040, 0.020),
     (0.010, 0.030), (0.0, 0.020)),
    # Round-up case (reviewer-required): _range_to_slice's explicit
    # branch ROUNDS range endpoints to the nearest cell, so an aperture
    # can snap ABOVE the declared width too, not only below it (as every
    # other SNAP case above does). domain y is intentionally larger than
    # the port's declared y_range so the range's upper edge rounds up
    # into the extra room: 0.02286/0.001 = 22.86 -> rounds to the 23rd
    # cell -> rasterized 23.000 mm > declared 22.860 mm.
    # port_aperture_snap is DIRECTION-AGNOSTIC by construction --
    # `_check_waveguide_port_aperture_snap` fires on
    # ``declared != rasterized``, never on the sign of the difference --
    # so this is exercised as an additional SNAP case rather than a new
    # table row.
    ("wr90_roundup_dx1_y_only", 1e-3, (0.10, 0.030, 0.010),
     (0.0, _A_WR90), None),
]

_TABLE = {
    # classification -> whether a port_aperture_snap finding is required
    "EXACT": False,          # declared == rasterized
    "SNAP": True,            # declared != rasterized (either direction --
                             # see wr90_roundup above)
    "UNRASTERIZABLE": True,  # _range_to_slice rejects the range outright
}


def _classify(declared, rasterized, tol=1e-12):
    if rasterized is None:
        return "UNRASTERIZABLE"
    if abs(declared - rasterized) <= tol:
        return "EXACT"
    return "SNAP"


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_every_port_pair_is_classified(case):
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
        declared = float(rng[1] - rng[0]) if rng is not None else float(sim._domain[ai])
        try:
            slc, _ = sim._range_to_slice(
                rng, sim._domain[ai], grid.dx, n_axis, grid.axis_pads[ai])
        except ValueError:
            rasterized = None
        else:
            rasterized = float((slc[1] - slc[0] - 1) * grid.dx)
        spans[ax] = (declared, rasterized)
    codes = _codes(sim)
    for axis, (declared, rasterized) in sorted(spans.items()):
        kind = _classify(declared, rasterized)
        assert kind in _TABLE, (
            f"{_id}/{axis}: unclassified span pair "
            f"declared={declared} rasterized={rasterized} -- extend "
            f"the classification table or the surface grew a new shape"
        )
        if _TABLE[kind]:
            expect = ("port_aperture_unrasterizable"
                      if kind == "UNRASTERIZABLE" else "port_aperture_snap")
            assert expect in codes, (
                f"{_id}/{axis} classified {kind} "
                f"(declared={declared*1e3:.4f} rasterized={rasterized} mm) "
                f"but preflight did not emit {expect}; "
                f"codes={codes!r}"
            )
    if all(_classify(*spans[ax]) == "EXACT" for ax in spans):
        assert "port_aperture_snap" not in codes, (
            f"{_id}: every axis EXACT but preflight fired a snap finding; "
            f"codes={codes!r}"
        )
