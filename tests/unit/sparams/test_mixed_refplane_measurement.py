"""Issue #498 / #517 — tests for the mixed-lane reference-plane measurement.

Four lanes, all cheap (pure NumPy, or ``num_periods <= 4``):

* **T3 / evaluators** — the F1 / F2 / F3 verdict functions on SYNTHETIC
  planted data. Every falsifier must be resolvable BOTH ways (each declared
  side reachable) and must return ``UNRESOLVED`` when its own precondition
  fails: the ring-down witness at or above -40 dB, or its discriminator below
  its own budget. ``NON_DISCRIMINATING`` is a legal, distinct outcome.
* **T3b / normalization identity** — ``|b_msl| == |out| / sqrt(Re Zc)`` on a
  synthetic single-travelling-wave line. Written against the WRONG
  ``|out|^2/(4 Re Zc)`` form first: that form under-reads by exactly 4 in
  power, and this test is what turns the corrected form green.
* **T1 / domain + absorber guard** — the mixed-lane reference-plane path must
  REJECT a plane that lands outside the declared domain, or within a face's
  absorber thickness of it, with a message class DISTINCT from "reach past
  another port".
* **T2 / crossing guard walks MSL probe ladders** — ``n_probes=5`` trips (the
  2N plane at 3.60 mm vs the last rung at 3.44 mm) and ``n_probes=3`` is
  silent at ``num_periods=4``.
* **T2b / the ladder walked is the RESOLVED one** — an AUTO probe spacing is
  re-derived at driver time (``_resolve_msl_auto_offsets``, #469/#681) and
  can be wider OR narrower than the registration value. The guard must trip
  when only the resolved ladder crosses the plane zone, and stay silent when
  only the registration ladder does. Both fixtures self-verify their premise
  (registered vs resolved rung indices) before the call.
* **Driver bookkeeping** — ``compute_mixed_s_matrix`` installs the resolved
  entries for each run, and ``compute_msl_s_matrix``'s rebuilt entries keep
  ``n_probes`` (it used to revert silently to the dataclass default 5).

The guard tests that raise never reach the solve (the guard runs while the
port sources are assembled), so they run in the default fast suite; only
the two tests that complete a solve carry ``@pytest.mark.slow``.

FAIL-BEFORE-FIX (measured on this worktree, pristine ``HEAD`` = 3038f845 with
``rfx/api/_execute.py`` restored, same driver script)::

    T1 (direction='+x', n_probes=3): NO RAISE (silent)
    T2 (direction='-x', n_probes=5): NO RAISE (silent)
    CONTROL (direction='-x', n_probes=3): NO RAISE (silent)

and after the guards::

    T1 (direction='+x', n_probes=3): ValueError -> reference_plane_cells:
        reference plane slot 1 ... lands at index 13 on axis 0 (x), within the
        absorber thickness of a declared-domain face ...
    T2 (direction='-x', n_probes=5): ValueError -> reference_plane_cells: the
        reference planes ... reach past another port at (0.0055, 0.0015, 0.0)
        — its MSL de-embedding probe 4 at x = 0.00344 m (index 51) ...
    CONTROL (direction='-x', n_probes=3): NO RAISE (silent)

Nothing here pins a lumped/wire diagonal, moves a tolerance, or touches a
reference — see §10 of
``docs/design_notes/mixed_refplane_predeclaration.md``.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_DRIVER = REPO / "scripts" / "diagnostics" / "mixed_refplane_measurement.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("_mixed_refplane_driver", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_driver()

# ===========================================================================
# T3 — F1 evaluator, every declared side reachable on planted data
# ===========================================================================

# ===========================================================================
# T3 — F2 evaluator, both sides + the two non-discriminating outcomes
# ===========================================================================

# ===========================================================================
# T3 — F3 evaluator
# ===========================================================================

# ===========================================================================
# T3b — the normalization identity, pure NumPy (fail-first against ÷4)
# ===========================================================================

def test_b_msl_equals_out_over_sqrt_zc_on_a_single_travelling_wave():
    """``refplane_split``'s 0.5 and ``_b_msl``'s 2 already cancel.

    Plant a pure -x-travelling wave on a real line: V = V-, I = -V-/Z. Then
    ``out`` at a ``-x``-outboard plane is exactly V- and
    ``b_msl = V-/sqrt(Z)``, so ``|b_msl| == |out|/sqrt(Re Zc)`` EXACTLY —
    and NOT ``|out|/(2 sqrt(Re Zc))``, the form that under-reads by 4 in
    power.
    """
    from rfx.probes.refplane import refplane_split

    z = 47.89479996289313
    v_minus = np.array([1.0 + 0.5j, -0.3 + 2.0j, 4.0 - 1.0j])
    v = v_minus
    i = -v_minus / z
    # outboard_sign = +1 (a "-x" port pushes its planes toward +x), so the
    # OUTGOING wave at the plane is w_plus = 0.5*(v + z*i)... which is zero
    # for a purely -x-travelling wave. Use the -x-outboard orientation, where
    # `outgoing` is w_minus = 0.5*(v - z*i) = V-.
    out, inc = refplane_split(v, i, z, outboard_sign=-1)
    np.testing.assert_allclose(out, v_minus, rtol=0, atol=1e-12)

    b_msl = m.b_msl_probe0(v, i, z)
    np.testing.assert_allclose(np.abs(b_msl), np.abs(v_minus) / np.sqrt(z),
                               rtol=1e-12)

    correct = np.abs(out) / np.sqrt(z)
    np.testing.assert_allclose(np.abs(b_msl), correct, rtol=1e-12)

    # The reviewer's ÷4-in-power form is wrong by exactly 2 in amplitude.
    wrong = np.abs(out) / (2.0 * np.sqrt(z))
    ratio = np.abs(b_msl) / wrong
    np.testing.assert_allclose(ratio, np.full_like(ratio, 2.0), rtol=1e-12)
    np.testing.assert_allclose((np.abs(b_msl) / wrong) ** 2,
                               np.full_like(ratio, 4.0), rtol=1e-12)


def test_refplane_split_half_is_already_inside():
    """Guards the premise of the identity above against a silent refactor."""
    from rfx.probes.refplane import refplane_split
    v = np.array([2.0 + 0.0j])
    i = np.array([0.0 + 0.0j])
    out, inc = refplane_split(v, i, 50.0, outboard_sign=+1)
    assert out[0] == pytest.approx(1.0)      # 0.5 * v, not v
    assert inc[0] == pytest.approx(1.0)


# ===========================================================================
# Report-only guards
# ===========================================================================

# ===========================================================================
# T1 / T2 — plumbing guards on the real mixed lane (num_periods = 4)
# ===========================================================================

def _run_mixed(direction: str, n_probes: int, num_periods: float = 4.0):
    """One instrumented two-drive call; returns the result or raises."""
    import contextlib
    import io

    sim = m.build_fixture(feed_direction=direction, n_probes=n_probes,
                          register_extra_flux=False)
    with m.refplane_instrumentation(m._REFPLANE_N):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return sim.compute_mixed_s_matrix(
                    freqs=np.linspace(1e9, 4e9, 5),
                    num_periods=num_periods, skip_preflight=True,
                )


def test_t1_mixed_refplane_path_rejects_absorber_adjacent_planes():
    """T1: ``direction="+x"`` puts slot 1 at index 13, six cells from the
    declared-domain lo face (index 8) inside the 8-cell absorber margin.

    On ``main`` this built SILENTLY (fail-before-fix evidence in the module
    docstring). The message class must be DISTINCT from the crossing guard's
    "reach past another port" — the two failures have different remedies.
    """
    with pytest.raises(ValueError) as ei:
        _run_mixed("+x", m._N_PROBES)
    msg = str(ei.value)
    assert "reference_plane_cells" in msg
    assert "absorber thickness of a declared-domain face" in msg or \
           "OUTSIDE the declared domain" in msg
    assert "reach past another port" not in msg


def test_t2_crossing_guard_walks_msl_probe_ladders_n_probes_5_trips():
    """T2 (bookkeeping): the default ``n_probes=5`` ladder's last rung is at
    x = 3.44 mm (index 51), INBOARD of the 2N plane at 3.60 mm (index 53).

    The port position alone (index 77) can never see this, which is why the
    guard walks the port's probe ladder. Message class stays "reach past
    another port".
    """
    with pytest.raises(ValueError) as ei:
        _run_mixed(m._FEED_DIRECTION, 5)
    msg = str(ei.value)
    assert "reach past another port" in msg
    assert "MSL de-embedding probe" in msg
    assert "0.00344" in msg          # the offending rung, named


@pytest.mark.slow
def test_t2_crossing_guard_silent_at_n_probes_3():
    """The declared deviation: ladder 4.72 / 4.40 / 4.08 mm, all outboard of
    the 2N plane. Must run clean — and must actually register the planes."""
    sim = m.build_fixture(feed_direction=m._FEED_DIRECTION,
                          n_probes=m._N_PROBES, register_extra_flux=False)
    import contextlib
    import io
    with m.refplane_instrumentation(m._REFPLANE_N) as cap:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                result = sim.compute_mixed_s_matrix(
                    freqs=np.linspace(1e9, 4e9, 5), num_periods=4.0,
                    skip_preflight=True)
    assert result is not None
    # B1 positive control: planes on BOTH drives, and the MSL-driven run
    # carries NO lw source.
    assert len(cap.positive_control) == 2
    assert all(pc["n_refplane_specs"] == 2 for pc in cap.positive_control)
    assert [pc["lw_excite_flags"] for pc in cap.positive_control] == \
        [[True], [False]]
    assert cap.positive_control[1]["drive_idx"] == m._NO_LW_DRIVE_SENTINEL


# ===========================================================================
# T2b — the crossing guard walks the RESOLVED ladder, not the registration
# ===========================================================================

class _Abort(Exception):
    """Raised by a capturing stub to stop a driver before its solve."""


def _build_with_ladder(*, freq_max, n_probes, n_probe_offset,
                       n_probe_spacing):
    """The lane fixture with the microstrip port's ladder arguments free.

    ``n_probe_spacing=None`` registers the conservative short default and
    lets ``_resolve_msl_auto_offsets`` re-derive it at driver time; on this
    5.5 mm feed the solve FLOORS the spacing at 2 cells for ``freq_max`` =
    5 GHz (λ_g/4 clearance exceeds the feed) and WIDENS it for 20 GHz.
    """
    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources.sources import GaussianPulse

    lx, ly, lz = m._DOMAIN
    sim = Simulation(
        freq_max=freq_max, domain=(lx, ly, lz), dx=m._DX,
        cpml_layers=m._CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=m._EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, m._H_SUB)), material="sub")
    y_c = ly / 2.0
    # 35 um foil -> a SHEET on the laminate face (#931 §1.3). The driver
    # script this fixture borrows its constants from still runs the
    # bisecting mesh (h_sub/dx = 254/80 = 3.175), so the sheet lands on
    # node 3 = 240 um, 14 um below the face it is drawn on; the contract
    # reports that offset rather than absorbing it. Redrawing the driver
    # on-lattice belongs with the driver (scripts/diagnostics), not here.
    sim.add(Box((0.0, y_c - m._W_TRACE / 2, m._H_SUB),
                (lx, y_c + m._W_TRACE / 2, m._H_SUB)), material="pec")
    sim.add_port(position=(m._X_FEED, y_c, 0.0), component="ez",
                 impedance=50.0, extent=m._H_SUB, direction=m._FEED_DIRECTION, terminates=1)
    sim.add_msl_port(position=(m._X_MSL, y_c, 0.0), width=m._W_TRACE,
                     height=m._H_SUB, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=freq_max / 2, bandwidth=0.5),
                     n_probe_offset=n_probe_offset,
                     n_probe_spacing=n_probe_spacing, n_probes=n_probes)
    return sim


def _ladder_indices(grid, entry) -> list[int]:
    """Line-axis indices of an entry's N-probe ladder, placed as the drivers
    place it (``msl_probe_x_coords_n`` on the entry's own offset/spacing)."""
    from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

    x_feed, y_c, z_lo = entry.position
    port = MSLPort(feed_x=x_feed, y_lo=y_c - entry.width / 2,
                   y_hi=y_c + entry.width / 2, z_lo=z_lo,
                   z_hi=z_lo + entry.height, direction=entry.direction,
                   impedance=entry.impedance, excitation=entry.waveform)
    xs = msl_probe_x_coords_n(grid, port, n_probes=entry.n_probes,
                              n_offset_cells=entry.n_probe_offset,
                              n_spacing_cells=entry.n_probe_spacing)
    return [int(grid.position_to_index((x, y_c, z_lo))[0]) for x in xs]


def _plane_zone(grid, sim, n_cells=m._REFPLANE_N):
    """``(i_port, i_far)`` of the lumped port's plane zone, as the guard
    defines it for a ``"-x"`` feed (planes pushed toward +x)."""
    pe = sim._ports[0]
    i_port = int(grid.position_to_index(pe.position)[0])
    return i_port, i_port + 2 * n_cells


def _registered_and_resolved(sim):
    from rfx.api._sparams import _resolve_msl_auto_offsets
    grid = sim._build_grid()
    registered = sim._msl_ports[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resolved = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), grid)[0]
    return grid, registered, resolved


def _crosses(idx, zone):
    i_port, i_far = zone
    return any(i_port < i <= i_far for i in idx)


def test_t2b_guard_trips_when_only_the_resolved_ladder_crosses():
    """Auto spacing WIDENED past the plane zone (#681): registration
    ``10/3`` -> rungs 67..55 (all outboard of the 2N plane at 53); resolved
    ``10/7`` -> rungs 67, 60, 53, 46, 39 (three inside). A guard reading the
    registration would build this silently; the run would probe inside the
    zone."""
    sim = _build_with_ladder(freq_max=20e9, n_probes=5, n_probe_offset=10,
                             n_probe_spacing=None)
    grid, reg, res = _registered_and_resolved(sim)
    zone = _plane_zone(grid, sim)
    reg_idx, res_idx = _ladder_indices(grid, reg), _ladder_indices(grid, res)
    # Premise, stated in indices so a resolver change is loud here first.
    assert res.n_probe_spacing > reg.n_probe_spacing
    assert not _crosses(reg_idx, zone), (reg_idx, zone)
    assert _crosses(res_idx, zone), (res_idx, zone)

    import contextlib
    import io
    with pytest.raises(ValueError) as ei:
        with m.refplane_instrumentation(m._REFPLANE_N):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    sim.compute_mixed_s_matrix(
                        freqs=np.linspace(1e9, 4e9, 5), n_steps=4,
                        skip_preflight=True)
    msg = str(ei.value)
    assert "reach past another port" in msg
    assert "MSL de-embedding probe" in msg
    # The rung it names is a RESOLVED rung inside the zone, and no
    # registration rung is inside the zone at all.
    named = int(msg.split("(index ")[1].split(")")[0])
    assert named in res_idx and _crosses([named], zone)
    assert named not in reg_idx
    # The driver restored the registration on the way out.
    assert sim._msl_ports[0].n_probe_spacing == reg.n_probe_spacing


def test_t2b_guard_silent_when_only_the_registration_ladder_crosses():
    """The converse: on the lane's own 5 GHz board the auto spacing FLOORS
    at 2 cells (the λ_g/4 clearance exceeds the 5.5 mm feed), so the
    registration ladder ``10/12`` -> 67, 55, 43, 31, 19 crosses the zone while
    the resolved ``10/2`` -> 67..59 does not. A guard reading the
    registration would reject a run that never probes inside the zone."""
    sim = _build_with_ladder(freq_max=m._FREQ_MAX, n_probes=5,
                             n_probe_offset=10, n_probe_spacing=None)
    grid, reg, res = _registered_and_resolved(sim)
    zone = _plane_zone(grid, sim)
    reg_idx, res_idx = _ladder_indices(grid, reg), _ladder_indices(grid, res)
    assert res.n_probe_spacing < reg.n_probe_spacing
    assert _crosses(reg_idx, zone), (reg_idx, zone)
    assert not _crosses(res_idx, zone), (res_idx, zone)

    # The guard alone, on the registered entries: must not raise.
    from rfx.api._execute import _refplane_reject_msl_ladder_in_plane_zone
    i_port, i_far = zone
    _refplane_reject_msl_ladder_in_plane_zone(
        sim, grid, sim._ports[0], line_axis=0, outboard_sign=+1,
        i_port=i_port, i_far=i_far)

    # And the whole mixed lane, with a 4-step record so the guard is the
    # only thing that could stop it: must build and run.
    import contextlib
    import io
    with m.refplane_instrumentation(m._REFPLANE_N) as cap:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                result = sim.compute_mixed_s_matrix(
                    freqs=np.linspace(1e9, 4e9, 5), n_steps=4,
                    skip_preflight=True)
    assert result is not None
    assert [pc["n_refplane_specs"] for pc in cap.positive_control] == [2, 2]


def test_mixed_driver_installs_the_resolved_ladder_for_each_run():
    """``compute_mixed_s_matrix`` rebuilds each run's ``self._msl_ports``
    from the RESOLVED entries (the ladder ``probe_xs`` is built from), not
    from the raw registration — captured at the solver entry and aborted
    before any solve."""
    from rfx import Simulation

    sim = _build_with_ladder(freq_max=m._FREQ_MAX, n_probes=5,
                             n_probe_offset=10, n_probe_spacing=None)
    grid, reg, res = _registered_and_resolved(sim)
    assert res.n_probe_spacing != reg.n_probe_spacing      # discriminating

    seen: list = []
    orig = Simulation._forward_from_materials

    def capture(self, *a, **kw):
        seen.append(list(self._msl_ports))
        raise _Abort

    Simulation._forward_from_materials = capture
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(_Abort):
                sim.compute_mixed_s_matrix(freqs=np.linspace(1e9, 4e9, 5),
                                           n_steps=4, skip_preflight=True)
    finally:
        Simulation._forward_from_materials = orig
    assert len(seen) == 1 and len(seen[0]) == 1
    installed = seen[0][0]
    assert (installed.n_probe_offset, installed.n_probe_spacing,
            installed.n_probes) == (res.n_probe_offset, res.n_probe_spacing,
                                    res.n_probes)
    # Restored on exit: the registration is untouched.
    assert sim._msl_ports[0] is reg


def test_msl_driver_rebuilt_entries_keep_n_probes():
    """``compute_msl_s_matrix`` rebuilds ``_MSLPortEntry`` per driven run;
    the rebuild used to omit ``n_probes`` (silently back to the dataclass
    default 5). Register a two-port microstrip thru with ``n_probes=3`` and
    read the entries at ``run()``, aborting before any solve."""
    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources.sources import GaussianPulse

    lx, ly, lz = m._DOMAIN
    sim = Simulation(
        freq_max=m._FREQ_MAX, domain=(lx, ly, lz), dx=m._DX,
        cpml_layers=m._CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=m._EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, m._H_SUB)), material="sub")
    y_c = ly / 2.0
    # 35 um foil -> a SHEET on the laminate face (#931 §1.3). The driver
    # script this fixture borrows its constants from still runs the
    # bisecting mesh (h_sub/dx = 254/80 = 3.175), so the sheet lands on
    # node 3 = 240 um, 14 um below the face it is drawn on; the contract
    # reports that offset rather than absorbing it. Redrawing the driver
    # on-lattice belongs with the driver (scripts/diagnostics), not here.
    sim.add(Box((0.0, y_c - m._W_TRACE / 2, m._H_SUB),
                (lx, y_c + m._W_TRACE / 2, m._H_SUB)), material="pec")
    pulse = GaussianPulse(f0=2.5e9, bandwidth=0.5)
    for x, direction in ((m._X_FEED, "+x"), (m._X_MSL, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=m._W_TRACE,
                         height=m._H_SUB, direction=direction, impedance=50.0,
                         waveform=pulse, n_probe_offset=10, n_probe_spacing=4,
                         n_probes=3)
    assert [pe.n_probes for pe in sim._msl_ports] == [3, 3]

    seen: list = []

    def fake_run(*a, **kw):
        seen.append([(pe.name, pe.n_probes, pe.n_probe_offset,
                      pe.n_probe_spacing) for pe in sim._msl_ports])
        raise _Abort

    sim.run = fake_run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_Abort):
            sim.compute_msl_s_matrix(freqs=np.linspace(1e9, 4e9, 5),
                                     num_periods=1.0)
    assert len(seen) == 1
    assert [row[1] for row in seen[0]] == [3, 3], seen
    assert [(row[2], row[3]) for row in seen[0]] == [(10, 4), (10, 4)]
