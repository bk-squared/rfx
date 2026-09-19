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
the REALIZED PEC edges along the port's own transverse line
(``guide_source="pec_walls"``; under the lattice ownership contract a
volume's drawn face IS a wall, so a 40 mm gap measures 40 mm — it read
42 mm while the far face of a body was never zeroed), or from the domain when the axis' two
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
from rfx.sources.waveguide_port import cutoff_frequency
import rfx.preflight.waveguide as wgpf
from tests._realized_geometry import assert_wall_planes, realized

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


def _transverse_pec(direction):
    """CPML on the launch axis, PEC on the two transverse ones.

    The same boundary shape as :func:`_pec_walls`, written for an arbitrary
    launch normal so the y- and z-normal fixtures are the x-normal one with
    the axes rotated and nothing else changed.
    """
    normal = direction[1]
    kw = {}
    for ax in "xyz":
        kw[ax] = "cpml" if ax == normal else Boundary(lo="pec", hi="pec")
    return BoundarySpec(**kw)


def _build(*, dx, domain, y_range=None, z_range=None, freqs, f0=None,
           mode_profile=None):
    sim = Simulation(freq_max=12e9, domain=domain, dx=dx,
                      boundary=_pec_walls(), cpml_layers=_CPML_LAYERS)
    kw = {}
    if y_range is not None:
        kw["y_range"] = y_range
    if z_range is not None:
        kw["z_range"] = z_range
    if mode_profile is not None:
        kw["mode_profile"] = mode_profile
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


def test_sub_aperture_walls_are_realized_where_they_are_drawn():
    """Build-time witness (no solve) for the number the finding must quote.

    The two PEC Boxes are VOLUMES (§1.2): they own the cells their centres
    fall in and realize a tangential wall on BOTH drawn faces, so the
    lower block's inner wall is the y node at 40 mm and the upper block's
    is the node at 80 mm — 20 cells, 40.0 mm, exactly the declared gap.
    Before #931 the far face of a body was never a wall, the metal ended
    one node short on each side, and the same geometry measured 42 mm (the
    #868 class). Read here from ``realized_pec_edge_masks`` so the
    expectation below is anchored on the realization, not on prose.
    """
    sim = _sub_aperture_sim(jnp.linspace(4.5e9, 7.0e9, 3), f0=6.0e9)
    grid = sim._build_grid()
    pad = grid.axis_pads[1]
    assert_wall_planes(
        sim, 1,
        expected_planes=list(range(pad, pad + 21)) + list(range(pad + 40, pad + 61)),
        what="sub-aperture guide walls")
    inner_lo, inner_hi = pad + 20, pad + 40
    assert (inner_hi - inner_lo) * grid.dx == pytest.approx(0.040, rel=1e-12), (
        "the drawn 40 mm gap must be the realized one")


def test_sub_aperture_guide_is_measured_from_the_pec_walls():
    """The interior PEC Boxes leave a 40 mm guide inside a 120 mm domain.
    The cutoffs the finding quotes must come from those walls.

    #931: the guide is the distance between the REALIZED wall planes, and
    a volume's drawn face is a wall, so the number is the drawn 40.0000 mm
    — not the 42.0000 mm the pre-#931 rule measured by treating the
    outermost METAL CELL as the wall (the far face was never zeroed, so
    the mask's last occupied index sat one node inside the real wall).
    fc_TE20 = c / 40.0 mm = 7.495 GHz, threshold 6.745 GHz; the band goes
    to 7.0 GHz so the advisory still has something to catch. The
    domain-extent version reported 120 mm -> fc_TE20 = 2.498 GHz.

    This assertion was committed as ``xfail(strict=True)`` while
    preflight's ``_port_transverse_spans`` still measured the guide from
    the primal CELL mask and read 42.0000 mm / 7.138 GHz / 6.424 GHz.  The
    migration landed with the preflight group's merge and the marker fired
    as an XPASS(strict) on 2026-09-07, so it is gone.  Not one number below
    changed when it came off — the assertions are the contract's, exactly
    as they were written before the consumer moved, which is the whole
    point of pre-declaring the falsifier.
    """
    sim = _sub_aperture_sim(jnp.linspace(4.5e9, 7.0e9, 3), f0=6.0e9)
    ev = [i for i in _issues(sim)
          if getattr(i, "code", None) == "port_evanescent"]
    assert ev, "7.0 GHz exceeds 0.90 x fc_TE20 on the walled guide"
    text = str(ev[0])
    assert "pec_walls" in text, (
        f"the guide must be measured on the realized wall planes along the "
        f"port's transverse line; got {text!r}"
    )
    assert "40.0000" in text and "7.495" in text and "6.745" in text, (
        f"expected the wall-measured 40.0000 mm guide (fc_TE20 7.495 GHz, "
        f"threshold 6.745 GHz); got {text!r}"
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
# 2b. #1101: the advisory must name the cutoff's REAL source, per profile.
#
# The message used to say "the solve builds its mode template and cutoff
# from <the width _range_to_slice reports>" on every port. That is true on
# ``mode_profile="analytic"`` and false on the API default ``"discrete"``,
# where ``init_waveguide_port`` takes ``f_c`` from the discrete eigenvalue
# of the aperture's own CELL widths. Both fixtures below are the SAME
# geometry -- WR-90 declared, dx = 1 mm, no explicit range, so the reported
# width is the declared 22.860 mm and the realized aperture is 23 cells --
# and they differ only in the profile, which is what makes the two messages
# a contrast rather than two separate claims.
#
# Build-only: ``preflight()`` never steps time, and neither does the
# ``_build_waveguide_port_config`` call these tests use as the oracle. The
# oracle is the BUILT config, not a formula re-derived here: a formula
# copied into a test drifts with the builder in the same direction as the
# message would, which is exactly the tautology #1101 was.
# --------------------------------------------------------------------------

_SNAP_DX = 1e-3
_SNAP_DOMAIN = (0.10, _A_WR90, _B_WR90)


def _snap_sim(mode_profile):
    return _build(dx=_SNAP_DX, domain=_SNAP_DOMAIN,
                  freqs=jnp.asarray([9e9]), f0=9e9,
                  mode_profile=mode_profile)


def _built_cfg(sim):
    """The port config the RUN builds, read on the same grid preflight uses."""
    grid = sim._build_grid()
    entry = sim._waveguide_ports[0]
    return grid, sim._build_waveguide_port_config(
        entry, grid, jnp.asarray(entry.freqs), 1)


def _snap_rows(sim):
    rows = [str(i) for i in _issues(sim)
            if getattr(i, "code", None) == "port_aperture_snap"]
    assert rows, "the WR-90-at-dx=1mm fixture must snap on both axes"
    return rows


def _ghz(hz):
    return f"{float(hz) / 1e9:.6f}"


def test_discrete_profile_row_quotes_the_built_cutoff_and_the_realized_aperture():
    """#1101, the default path: the cutoff is the realized aperture's.

    Pins the INVARIANT -- the row quotes the number the builder produced,
    and names the realized aperture as its source -- not a value. The
    declared-width analytic cutoff is asserted to be a DIFFERENT number, so
    the check cannot pass on a message that quotes the declared one.
    """
    sim = _snap_sim("discrete")
    grid, cfg = _built_cfg(sim)
    declared_analytic = cutoff_frequency(_A_WR90, _B_WR90, 1, 0)
    assert _ghz(cfg.f_cutoff) != _ghz(declared_analytic), (
        "fixture no longer separates the two cutoffs, so this test could "
        f"pass on the wrong one: both read {_ghz(cfg.f_cutoff)} GHz"
    )
    n_cells_y = int(cfg.u_hi - cfg.u_lo)

    for row in _snap_rows(sim):
        assert "realized aperture" in row.lower(), (
            f"the discrete profile builds the cutoff from the realized "
            f"aperture and the row must say so; got {row!r}"
        )
        assert f"f_cutoff = {_ghz(cfg.f_cutoff)} GHz" in row, (
            f"the row must quote the BUILT cutoff "
            f"{_ghz(cfg.f_cutoff)} GHz; got {row!r}"
        )
        assert f"the declared width would give {_ghz(declared_analytic)} GHz" in row, (
            f"the row must still offer the declared-width analytic value "
            f"for comparison; got {row!r}"
        )
    y_row = [r for r in _snap_rows(sim) if "y-width" in r][0]
    assert f"{n_cells_y} cells" in y_row, (
        f"the y row must quote the realized aperture's own cell count "
        f"({n_cells_y}); got {y_row!r}"
    )


def test_analytic_profile_row_says_the_cutoff_comes_from_the_declared_width():
    """#1101, the other path: ``mode_profile="analytic"`` really does read
    ``cutoff_frequency(port.a, port.b, m, n)``, and with no explicit range
    ``port.a`` IS the declared width. The pre-#1101 sentence was right here,
    so the row keeps saying so -- and quotes the same three numbers, so a
    reader can tell the two paths apart without rebuilding the port."""
    sim = _snap_sim("analytic")
    grid, cfg = _built_cfg(sim)
    declared_analytic = cutoff_frequency(_A_WR90, _B_WR90, 1, 0)
    assert _ghz(cfg.f_cutoff) == _ghz(declared_analytic), (
        "with no explicit range the analytic profile's cutoff IS the "
        f"declared width's: built {_ghz(cfg.f_cutoff)} GHz vs "
        f"{_ghz(declared_analytic)} GHz"
    )
    for row in _snap_rows(sim):
        assert "mode_profile='analytic'" in row, (
            f"the row must name the profile whose behaviour it describes; "
            f"got {row!r}"
        )
        assert "declared" in row and "REALIZED aperture" not in row, (
            f"on the analytic profile the cutoff does NOT come from the "
            f"realized aperture; got {row!r}"
        )
        assert f"f_cutoff = {_ghz(cfg.f_cutoff)} GHz" in row, row


def test_the_two_profiles_do_not_get_the_same_cutoff_sentence():
    """The whole of #1101 in one assertion: one message for two different
    builds is how the wrong narrative got pinned onto 12 gate rows."""
    discrete = _snap_rows(_snap_sim("discrete"))[0]
    analytic = _snap_rows(_snap_sim("analytic"))[0]
    assert discrete != analytic, (
        "the two profiles build the cutoff from different widths; one "
        f"sentence cannot describe both. Got {discrete!r}"
    )


def test_the_discrete_row_names_the_mode_and_who_picks_the_eigenvalue():
    """#1101 review P7. Two claims the row has to carry.

    The number is ONE mode's cutoff, so the row names that mode. And
    "built from the realized aperture" alone overstates the mechanism:
    ``_discrete_te_mode_profiles`` solves on the realized cell widths but
    SELECTS among the eigenvectors by overlap with the analytic profile
    built from the declared ``port.a``/``port.b``, which its own TE30/TE21
    comment says can decide the answer. Both halves belong in the clause.
    """
    sim = _snap_sim("discrete")
    _grid, cfg = _built_cfg(sim)
    m, n = tuple(cfg.mode_indices)
    for row in _snap_rows(sim):
        assert f"{cfg.mode_type}{m}{n} mode template" in row, (
            f"the row must name the mode its cutoff belongs to "
            f"({cfg.mode_type}{m}{n}); got {row!r}"
        )
        assert ("eigenvalues from the realized cells; the declared widths "
                "pick which eigenvalue") in row, (
            f"the row must not claim the realized aperture alone decides "
            f"the eigenvalue; got {row!r}"
        )


# --------------------------------------------------------------------------
# 2c. #1101 review P1: the discrete mode solve is a DENSE eigh of an
#     (nu*nv, nu*nv) matrix, so preflight must not run it on a large
#     aperture. Measured on this pod, two-port WR-90 preflight(strict=False):
#     0.07 s at dx = 1 mm (230 cells), 0.39 s at 0.5 mm (920), 4.79 s at
#     0.25 mm (3731) and 127.21 s / 10.7 GB at 0.125 mm (14823). The budget
#     is a committed constant; these tests pin the BEHAVIOUR on either side
#     of it, not the constant's value.
# --------------------------------------------------------------------------

class _BuilderMustNotRun(AssertionError):
    """Raised by the monkeypatched builder, so a call cannot pass silently."""


def _wr90_sim(dx, mode_profile="discrete"):
    """WR-90 declared at an arbitrary dx. Every dx below snaps on both axes."""
    return _build(dx=dx, domain=(0.10, _A_WR90, _B_WR90),
                  freqs=jnp.asarray([9e9]), f0=9e9,
                  mode_profile=mode_profile)


def _cells(sim):
    grid = sim._build_grid()
    entry = sim._waveguide_ports[0]
    spans = sim._port_transverse_spans(entry, grid, None)
    return (int(round(spans["y"]["rasterized"] / grid.dx))
            * int(round(spans["z"]["rasterized"] / grid.dx)))


def test_an_aperture_over_the_budget_never_calls_the_port_builder(monkeypatch):
    """The cost guard, pinned where it matters: the builder is not CALLED.

    Timing a test would be flaky; refusing to let the builder run at all is
    not. 0.125 mm is the 14823-cell / 127 s case from the table above.
    """
    sim = _wr90_sim(0.125e-3)
    assert _cells(sim) > wgpf.WAVEGUIDE_PREFLIGHT_DISCRETE_CELL_BUDGET, (
        "fixture no longer exceeds the budget, so this test proves nothing"
    )

    def _boom(self, *a, **k):
        raise _BuilderMustNotRun(
            "preflight built a port config for an over-budget aperture")

    monkeypatch.setattr(type(sim), "_build_waveguide_port_config", _boom)
    rows = _snap_rows(sim)
    for row in rows:
        assert "f_cutoff is not solved here" in row, row
        assert "analytic (realized aperture)" in row, row
        assert "O(dx^2)" in row, row
        assert (f"{wgpf.WAVEGUIDE_PREFLIGHT_DISCRETE_CELL_BUDGET}-cell budget"
                in row), row


def test_an_aperture_under_the_budget_still_reads_the_built_cutoff():
    """The other side of the same guard: under the budget nothing changed."""
    sim = _wr90_sim(0.5e-3)
    assert _cells(sim) <= wgpf.WAVEGUIDE_PREFLIGHT_DISCRETE_CELL_BUDGET
    _grid, cfg = _built_cfg(sim)
    for row in _snap_rows(sim):
        assert f"f_cutoff = {_ghz(cfg.f_cutoff)} GHz" in row, row
        assert "is not solved here" not in row, row


def test_the_analytic_profile_is_not_subject_to_the_budget():
    """``mode_profile="analytic"`` has no eigensolve — ``cutoff_frequency``
    and the profile helpers are closed form — so the budget must not refuse
    it. The same 0.125 mm aperture that the discrete path declines."""
    sim = _wr90_sim(0.125e-3, mode_profile="analytic")
    assert _cells(sim) > wgpf.WAVEGUIDE_PREFLIGHT_DISCRETE_CELL_BUDGET
    _grid, cfg = _built_cfg(sim)
    for row in _snap_rows(sim):
        assert f"f_cutoff = {_ghz(cfg.f_cutoff)} GHz" in row, row
        assert "is not solved here" not in row, row


# --------------------------------------------------------------------------
# 2d. #1101 review P4: both "could not read it" strings must be reachable.
# --------------------------------------------------------------------------

def test_a_builder_failure_falls_back_to_the_analytic_realized_value(monkeypatch):
    """``preflight(strict=False)`` COLLECTS findings and never crashes, so a
    builder that raises has to leave a row that says so rather than an
    exception. Reached in the wild when ``_build_waveguide_port_config``
    rejects measurement planes that leave the domain."""
    sim = _snap_sim("discrete")
    grid = sim._build_grid()
    entry = sim._waveguide_ports[0]
    spans = sim._port_transverse_spans(entry, grid, None)
    realized_analytic = cutoff_frequency(
        spans["y"]["rasterized"], spans["z"]["rasterized"], 1, 0)

    def _raise(self, *a, **k):
        raise ValueError("synthetic build failure")

    monkeypatch.setattr(type(sim), "_build_waveguide_port_config", _raise)
    rows = _snap_rows(sim)
    for row in rows:
        assert "f_cutoff could not be read here" in row, row
        assert "ValueError: synthetic build failure" in row, (
            f"the row must say WHY it could not read the cutoff; got {row!r}"
        )
        assert f"{_ghz(realized_analytic)} GHz, analytic (realized aperture)" in row, row


def test_a_port_with_one_unrasterizable_axis_asserts_nothing_about_the_cutoff():
    """The other fallback. With one transverse axis rejected by the compiler
    there is no (a, b) pair, so the row keeps the half it can measure and
    says nothing about the cutoff. Live on a committed shape: WR-90 declared
    at dx = 1 mm with EXPLICIT ranges — y rasterizes to 23.000 mm, wider than
    the 22.860 mm domain, which ``_range_to_slice`` rejects, while z still
    snaps 10.160 -> 10.000 mm."""
    sim = _build(dx=1e-3, domain=(0.10, _A_WR90, _B_WR90),
                 y_range=(0.0, _A_WR90), z_range=(0.0, _B_WR90),
                 freqs=jnp.asarray([9e9]), f0=9e9)
    codes = _codes(sim)
    assert "port_aperture_unrasterizable" in codes, codes
    rows = _snap_rows(sim)
    for row in rows:
        assert "the cutoff this port builds could not be read on this grid" in row, row
        assert "f_cutoff = " not in row, (
            f"nothing may be asserted about a cutoff that was not read; "
            f"got {row!r}"
        )


# --------------------------------------------------------------------------
# 2e. #1101 review P5: the (u, v) axis mapping, on all three launch normals.
#
# ``_build_waveguide_port_config`` assigns ``WaveguidePort.a`` to the u axis
# and ``.b`` to the v axis, per launch normal: x-normal -> (y, z), y-normal ->
# (x, z), z-normal -> (x, y). The advisory quotes a per-axis cell count, so a
# swapped mapping would print b's count against a's width. Every fixture above
# is x-normal; these two cover the other two normals with DIFFERENT counts on
# the two axes, so a swap cannot pass.
# --------------------------------------------------------------------------

_UV_CASES = [
    # (id, direction, domain, long axis, short axis)
    # The long transverse wall is 22.86 mm (23 cells at dx = 1 mm), the short
    # one 10.16 mm (11 cells); the launch axis is 100 mm.
    ("y_normal", "+y", (_A_WR90, 0.10, _B_WR90), "x", "z"),
    ("z_normal", "+z", (_A_WR90, _B_WR90, 0.10), "x", "y"),
]


@pytest.mark.parametrize("case", _UV_CASES, ids=[c[0] for c in _UV_CASES])
def test_each_axis_row_quotes_its_own_realized_cell_count(case):
    _id, direction, domain, long_ax, short_ax = case
    sim = Simulation(freq_max=12e9, domain=domain, dx=1e-3,
                     boundary=_transverse_pec(direction),
                     cpml_layers=_CPML_LAYERS)
    sim.add_waveguide_port(0.024, direction=direction, mode=(1, 0),
                           mode_type="TE", freqs=jnp.asarray([9e9]),
                           f0=9e9, name="p0")
    rows = {}
    for row in _snap_rows(sim):
        for ax in (long_ax, short_ax):
            if f"declared {ax}-width" in row:
                rows[ax] = row
    assert set(rows) == {long_ax, short_ax}, (
        f"{_id}: expected a snap row on both transverse axes, got "
        f"{sorted(rows)}"
    )
    # 22.860 mm -> 23 cells, 10.160 mm -> 11 cells at dx = 1 mm. Different
    # counts on the two axes is what makes a swapped mapping visible.
    assert "23 cells = 23.0000 mm" in rows[long_ax], (
        f"{_id}: the {long_ax} row must carry the 23-cell count of the "
        f"22.860 mm wall; got {rows[long_ax]!r}"
    )
    assert "11 cells = 11.0000 mm" in rows[short_ax], (
        f"{_id}: the {short_ax} row must carry the 11-cell count of the "
        f"10.160 mm wall; got {rows[short_ax]!r}"
    )


# --------------------------------------------------------------------------
# 2f. #1101 review: no cubic-cell assumption (repo engineering principle 2).
#
# A transverse span is ``cells * that axis' own cell size``, never
# ``cells * dx``. The uniform ``Grid`` carries only ``dx``, so on every grid
# that reaches this family today the per-axis read returns ``dx`` and nothing
# moves numerically -- that no-op is asserted below rather than assumed. The
# ``dy != dz`` case is exercised through a proxy grid, because the only grid
# class with per-axis sizes is ``NonUniformGrid`` and
# ``_check_waveguide_port_evanescent`` routes that to the declared-geometry
# lane before the spans are computed.
#
# What is deliberately NOT made axis-aware here: ``_range_to_slice``'s index
# math. ``_build_waveguide_port_config`` calls it with ``grid.dx`` on every
# axis, and preflight mirroring the builder is the whole point of this check.
# That cubic assumption lives in the builder and is out of #1101's scope.
# --------------------------------------------------------------------------

class _AnisotropicGridProxy:
    """A real grid with two transverse cell sizes bolted on.

    Everything except ``dy``/``dz`` delegates to the grid the simulation
    actually built, so the slices, pads and node counts under test are the
    committed ones and only the per-axis cell size is synthetic.
    """

    def __init__(self, grid, dy=None, dz=None):
        self._grid = grid
        if dy is not None:
            self.dy = dy
        if dz is not None:
            self.dz = dz

    def __getattr__(self, name):
        return getattr(self._grid, name)


def test_the_per_axis_cell_size_is_a_no_op_on_a_uniform_grid():
    """The uniform ``Grid`` has no ``dy``/``dz``, so the axis-aware read must
    return ``dx`` and every span must be bit-identical to the scalar form.
    This is what keeps the 12 pinned gate rows from moving."""
    sim = _snap_sim("discrete")
    grid = sim._build_grid()
    assert not hasattr(grid, "dy") and not hasattr(grid, "dz"), (
        "the uniform Grid grew per-axis cell sizes; re-check this no-op"
    )
    for ax in "xyz":
        assert wgpf._transverse_cell_size(grid, ax) == float(grid.dx)
    spans = sim._port_transverse_spans(sim._waveguide_ports[0], grid, None)
    for ax, rec in spans.items():
        ai = "xyz".index(ax)
        n_axis = (grid.nx, grid.ny, grid.nz)[ai]
        slc, _ = sim._range_to_slice(
            getattr(sim._waveguide_ports[0], f"{ax}_range"),
            sim._domain[ai], grid.dx, n_axis, grid.axis_pads[ai])
        assert rec["rasterized"] == float((slc[1] - slc[0] - 1) * grid.dx)
    for row in _snap_rows(sim):
        assert f"(dx={grid.dx * 1e3:.4f} mm)" in row, (
            f"on a cubic grid the row must still name dx; got {row!r}"
        )


def test_a_nonuniform_axis_size_is_read_per_axis_not_from_dx():
    """``getattr(grid, "dy", grid.dx)`` has to reach the span arithmetic.

    The proxy gives y cells twice dx and z cells three times dx, so a span
    still computed from ``dx`` would read the cubic number and fail here.
    """
    sim = _snap_sim("discrete")
    grid = sim._build_grid()
    dx = float(grid.dx)
    proxy = _AnisotropicGridProxy(grid, dy=2.0 * dx, dz=3.0 * dx)
    assert wgpf._transverse_cell_size(proxy, "x") == dx
    assert wgpf._transverse_cell_size(proxy, "y") == 2.0 * dx
    assert wgpf._transverse_cell_size(proxy, "z") == 3.0 * dx

    entry = sim._waveguide_ports[0]
    cubic = sim._port_transverse_spans(entry, grid, None)
    aniso = sim._port_transverse_spans(entry, proxy, None)
    for ax, factor in (("y", 2.0), ("z", 3.0)):
        assert aniso[ax]["rasterized"] == pytest.approx(
            cubic[ax]["rasterized"] * factor), (
            f"the {ax} span must scale with d{ax}, not stay on dx: "
            f"{aniso[ax]['rasterized']} vs {cubic[ax]['rasterized']}"
        )
        assert aniso[ax]["guide"] == pytest.approx(
            cubic[ax]["guide"] * factor), ax

    reading = wgpf._waveguide_port_cutoff_reading(sim, entry, proxy, aniso)
    assert reading is not None
    # The cell COUNT is unchanged -- the same slice, bigger cells -- which is
    # the thing a dx-based division would get wrong once the span moved.
    for ax in ("y", "z"):
        assert reading["cells"][ax] == int(
            round(cubic[ax]["rasterized"] / dx)), ax


def test_an_anisotropic_grid_names_the_axis_cell_size_it_used():
    """A row that computed its span from ``dy`` must not print ``dx=``."""
    sim = _snap_sim("discrete")
    grid = sim._build_grid()
    dx = float(grid.dx)
    proxy = _AnisotropicGridProxy(grid, dy=2.0 * dx, dz=3.0 * dx)
    entry = sim._waveguide_ports[0]
    spans = sim._port_transverse_spans(entry, proxy, None)

    import contextlib as _c
    import io as _io
    import warnings as _wmod
    with _c.redirect_stdout(_io.StringIO()):
        with _wmod.catch_warnings(record=True) as caught:
            _wmod.simplefilter("always")
            sim._check_waveguide_port_aperture_snap(proxy, None)
    rows = [str(r.message) for r in caught
            if getattr(r.message, "code", None) == "port_aperture_snap"]
    assert rows, "the anisotropic proxy must still snap on both axes"
    by_axis = {ax: [r for r in rows if f"declared {ax}-width" in r]
               for ax in ("y", "z")}
    for ax, factor in (("y", 2.0), ("z", 3.0)):
        assert by_axis[ax], f"no {ax} row"
        row = by_axis[ax][0]
        assert f"(d{ax}={factor * dx * 1e3:.4f} mm)" in row, (
            f"the {ax} row must name d{ax}, not dx; got {row!r}"
        )
        assert f"covers {spans[ax]['rasterized'] * 1e3:.4f} mm" in row, row


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
