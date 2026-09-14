"""The ``Simulation.compute_s_matrix`` dispatch table (issue #980 Phase 1).

``compute_s_matrix`` picks one of the per-family S-matrix calculators from the
port registrations and forwards ``**kwargs`` to it unchanged; ``s_matrix_lane``
answers the same question and returns the chosen method's NAME without running
anything. Two kinds of test here:

*Table tests* (the bulk, no FDTD). Every row of the documented table and every
error path is asserted on a registration-only ``Simulation`` -- ports
registered, nothing run. That is what ``s_matrix_lane`` exists for: it makes
the routing decision observable without paying for a solve, so a wrong route
is caught by a millisecond of test rather than by a physics run that returns
plausible-looking numbers from the wrong lane.

*Equivalence tests* (four legs, real FDTD). Routing correctly is not enough --
``compute_s_matrix(**kw)`` has to produce the SAME arrays as calling the
delegate with the same ``**kw``, or the dispatcher is a second, divergent
implementation. Each leg builds two identical simulations, drives one through
the dispatcher and one directly, and compares with ``np.array_equal`` -- bit
identity, no tolerance, the same gate ``docs/agent-memory/
development_methodology.md`` section 2.2 puts on code motion. The fixtures and
truncated kwargs are copied verbatim from
``tests/locks/test_sparams_split_bit_identity.py``, which selected them as the
smallest deterministic run that walks each whole path. Those records are
deliberately UNSETTLED and their |S| values are not calibrated numbers: this
file measures agreement between two call paths, never physics. The calibration
gates stay in each lane's own test module.

Warnings are suppressed in the equivalence legs for the same reason the lock
suppresses them -- the truncated records warn by design (settling witness), the
advisory behaviour is pinned in ``tests/unit/sparams`` already, and no physics
number is reported out of this file.
"""

from __future__ import annotations

import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse


# ---------------------------------------------------------------------------
# Registration-only builders, one per row of the table. Kept deliberately
# minimal: nothing here is run, so a builder only has to register the ports
# that decide the lane.
# ---------------------------------------------------------------------------

_WG_FREQS = np.linspace(4e9, 6e9, 6)
_COAX_BAND = jnp.asarray([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])
_MSL_FREQS = jnp.linspace(2e9, 18e9, 16)


def _waveguide_sim() -> Simulation:
    """``tests/_pec_short_advisory_fixture`` at dx = 2 mm, cpml 8.

    The same build ``tests/unit/sparams/test_sparam_passivity_guard.py`` and
    the #980 bit-identity lock drive; two ``add_waveguide_port`` calls, which
    is what the waveguide row needs.
    """
    from tests._pec_short_advisory_fixture import build

    return build(_WG_FREQS, dx=2e-3, cpml=8)


def _msl_sim() -> Simulation:
    """``_thru()`` of ``tests/unit/sparams/test_msl_passivity_enforcement.py``."""
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0., 0., 0.), (0.012, 0.008, 0.)), material="pec")
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


_MIXED_EPS_R = 3.66
_MIXED_H_SUB = 254e-6
_MIXED_W_TRACE = 600e-6
_MIXED_DX = _MIXED_H_SUB / 3.0


def _mixed_sim() -> Simulation:
    """Layer-1c smoke of ``tests/unit/sparams/test_mixed_port_sparam.py``."""
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_MIXED_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_MIXED_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _MIXED_H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _MIXED_W_TRACE / 2, _MIXED_H_SUB),
                (lx, y_c + _MIXED_W_TRACE / 2, _MIXED_H_SUB)), material="pec")
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_MIXED_H_SUB)
    sim.add_msl_port(position=(5.5e-3, y_c, 0.0), width=_MIXED_W_TRACE,
                     height=_MIXED_H_SUB, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
                     n_probe_offset=10, n_probe_spacing=4)
    return sim


def _coax_sim() -> Simulation:
    """``_coax_two_port_sim()`` of ``tests/unit/sparams/test_settling_witness.py``.

    ONE ``add_coaxial_port`` -- which is also the whole
    ``compute_coaxial_line_reflection`` registration. The two lanes are
    indistinguishable from here; that is the ambiguity this file pins.
    """
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


def _coax_msl_sim() -> Simulation:
    """``tests/_coax_msl_instrument_fixture`` -- one coax + one MSL port."""
    from tests._coax_msl_instrument_fixture import build_instrument_junction

    return build_instrument_junction()


def _lumped_only_sim() -> Simulation:
    """``_mixed_sim()`` minus its MSL port: the run-lane row."""
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(freq_max=5e9, domain=(lx, ly, lz), dx=_MIXED_DX,
                     cpml_layers=8, boundary="cpml")
    sim.add_port(position=(2e-3, ly / 2.0, 0.0), component="ez",
                 impedance=50.0, extent=_MIXED_H_SUB)
    return sim


def _empty_sim() -> Simulation:
    return Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01))


def _floquet_sim() -> Simulation:
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.02), boundary="cpml")
    sim.add_floquet_port(0.005, axis="z")
    return sim


def _tfsf_sim() -> Simulation:
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.02), boundary="cpml")
    sim.add_tfsf_source(f0=5e9)
    return sim


def _three_coax_sim() -> Simulation:
    sim = Simulation(freq_max=10.0e9, domain=(0.020, 0.020, 0.020),
                     boundary="pec")
    for z in (0.005, 0.010, 0.015):
        sim.add_coaxial_port((0.010, 0.010, z), face="top")
    return sim


def _waveguide_plus_msl_sim() -> Simulation:
    sim = _waveguide_sim()
    sim.add_msl_port(position=(0.020, 0.020, 0.0), width=0.0012,
                     height=0.0008, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="m1")
    return sim


def _bare_source_plus_msl_sim() -> Simulation:
    """``add_source()`` shares ``self._ports`` with real lumped ports.

    It registers a ``_PortEntry`` with ``impedance=0.0``, so a census that
    only counted ``len(self._ports)`` would route this to the mixed lane --
    which rejects it, because a bare source is not excite-gated and fires in
    every drive run.
    """
    sim = _msl_sim()
    sim.add_source(position=(0.006, 0.004, 0.0016), component="ez")
    return sim


# ---------------------------------------------------------------------------
# The table: every row that resolves to a lane.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder,expected", [
    (_waveguide_sim, "compute_waveguide_s_matrix"),
    (_msl_sim, "compute_msl_s_matrix"),
    (_coax_msl_sim, "compute_coax_msl_transition"),
    (_mixed_sim, "compute_mixed_s_matrix"),
    (_lumped_only_sim, "run(compute_s_params=True)"),
], ids=["waveguide", "msl", "coax+msl", "lumped+msl", "lumped-only"])
def test_s_matrix_lane_resolves_each_determinate_row(builder, expected):
    """No FDTD: registrations alone select the lane."""
    assert builder().s_matrix_lane() == expected


def test_lumped_only_lane_is_the_run_lane_string_not_a_method():
    """The lumped/wire family has no ``compute_*`` method at all.

    ``s_matrix_lane()`` still answers (the table stays total), but it names
    the run lane rather than a method that does not exist.
    """
    lane = _lumped_only_sim().s_matrix_lane()
    assert lane == "run(compute_s_params=True)"
    assert not hasattr(Simulation, lane)


# ---------------------------------------------------------------------------
# The ambiguous row: one coaxial port, two lanes, identical footprints.
# ---------------------------------------------------------------------------

def test_single_coaxial_port_is_ambiguous_and_is_not_guessed():
    """Both coaxial lanes require EXACTLY ONE ``add_coaxial_port()``.

    ``compute_coaxial_two_port`` mirrors that one registered port into a
    through-line internally rather than consuming a second registration, so
    its registration footprint is byte-for-byte the one
    ``compute_coaxial_line_reflection`` wants. The dispatcher must say so
    instead of picking the one whose result type happens to be wrong.
    """
    sim = _coax_sim()
    assert len(sim._coaxial_ports) == 1
    with pytest.raises(NotImplementedError) as exc:
        sim.s_matrix_lane()
    msg = str(exc.value)
    assert "compute_coaxial_line_reflection" in msg
    assert "compute_coaxial_two_port" in msg
    assert "lane=" in msg


@pytest.mark.parametrize("lane", [
    "compute_coaxial_line_reflection",
    "compute_coaxial_two_port",
])
def test_lane_override_resolves_the_coaxial_ambiguity(lane):
    assert _coax_sim().s_matrix_lane(lane=lane) == lane


def test_lane_override_must_agree_with_an_unambiguous_registration():
    """``lane=`` picks between two lanes that share a footprint.

    It is not a way to run the coaxial lane on a waveguide simulation: where
    the registrations already determine the answer, a disagreeing ``lane=``
    is a caller error, not an override of the port-family fence.
    """
    with pytest.raises(ValueError, match="contradicts the registrations"):
        _waveguide_sim().s_matrix_lane(lane="compute_coaxial_two_port")


def test_lane_override_agreeing_with_the_registration_is_accepted():
    sim = _waveguide_sim()
    assert sim.s_matrix_lane(lane="compute_waveguide_s_matrix") == (
        "compute_waveguide_s_matrix")


def test_unknown_lane_name_is_rejected_with_the_valid_set():
    with pytest.raises(ValueError) as exc:
        _coax_sim().s_matrix_lane(lane="compute_s_params")
    assert "compute_waveguide_s_matrix" in str(exc.value)


def test_deprecated_coaxial_lane_is_not_selectable():
    """``compute_coaxial_s_matrix`` is the DEPRECATED single-plane path.

    Its own docstring records measured, non-physical ``|S11|>1`` for a
    lossless short. The dispatcher never routes to it, and ``lane=`` cannot
    ask for it either -- a caller who really wants it calls it directly and
    sees its deprecation warning.
    """
    assert hasattr(Simulation, "compute_coaxial_s_matrix")
    with pytest.raises(ValueError) as exc:
        _coax_sim().s_matrix_lane(lane="compute_coaxial_s_matrix")
    assert "deprecated" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Error paths. Each asserts the TYPE and that the message names the method
# (or lane) the caller should actually use -- a diagnostic that only says
# "unsupported" sends the reader back to grepping the mixins.
# ---------------------------------------------------------------------------

def test_lumped_only_compute_points_at_the_run_lane():
    with pytest.raises(NotImplementedError) as exc:
        _lumped_only_sim().compute_s_matrix()
    msg = str(exc.value)
    assert "compute_s_params=True" in msg
    assert "res.s_params" in msg
    assert "n_steps" in msg


def test_lumped_only_compute_does_not_silently_run_the_solve(monkeypatch):
    """It must RAISE, not quietly call ``run()`` for the caller.

    ``run()`` returns a ``Result``, not an S-matrix result type, and its
    ``n_steps`` has no default -- substituting one would be inventing exactly
    the kind of default this dispatcher refuses to invent. Sentinel: if the
    dispatcher ever reaches ``run()``, this test reds with a RuntimeError
    instead of passing on the NotImplementedError.
    """
    def boom(self, *a, **kw):
        raise RuntimeError("RUN-LANE-ENTERED")

    monkeypatch.setattr(Simulation, "run", boom)
    with pytest.raises(NotImplementedError):
        _lumped_only_sim().compute_s_matrix()


def test_no_ports_lists_the_builders():
    with pytest.raises(ValueError) as exc:
        _empty_sim().s_matrix_lane()
    msg = str(exc.value)
    for builder in ("add_waveguide_port", "add_msl_port", "add_coaxial_port",
                    "add_port"):
        assert builder in msg, f"{builder} missing from the no-ports message"


def test_no_ports_from_compute_raises_the_same_valueerror():
    with pytest.raises(ValueError, match="add_waveguide_port"):
        _empty_sim().compute_s_matrix()


@pytest.mark.parametrize("builder,must_name", [
    (_floquet_sim, ["add_floquet_port", "no S-matrix calculator"]),
    (_tfsf_sim, ["add_tfsf_source", "not a port"]),
    (_three_coax_sim, ["add_coaxial_port x3", "EXACTLY ONE",
                       "compute_coaxial_s_matrix"]),
    (_waveguide_plus_msl_sim, ["add_waveguide_port x2", "add_msl_port x1",
                               "compute_waveguide_s_matrix",
                               "compute_msl_s_matrix"]),
    (_bare_source_plus_msl_sim, ["add_source", "compute_mixed_s_matrix"]),
], ids=["floquet", "tfsf", "3-coax", "waveguide+msl", "bare-source+msl"])
def test_unsupported_combinations_name_the_families_and_the_way_out(
        builder, must_name):
    with pytest.raises(NotImplementedError) as exc:
        builder().s_matrix_lane()
    msg = str(exc.value)
    for token in must_name:
        assert token in msg, f"{token!r} missing from:\n{msg}"


def test_three_coax_does_not_route_to_the_deprecated_lane():
    """The only method that accepts several coaxial ports is deprecated.

    Routing there would be the worst outcome of the whole dispatcher: a
    measured non-physical result reached by a convenience helper.
    """
    with pytest.raises(NotImplementedError) as exc:
        _three_coax_sim().compute_s_matrix()
    assert "DEPRECATED" in str(exc.value)


# ---------------------------------------------------------------------------
# Naming: the qualname rewrite, and whose name a bad keyword reports.
# ---------------------------------------------------------------------------

def test_dispatcher_methods_report_the_public_class_name():
    """``rfx/api/__init__.py`` rewrites ``_SparamMixin.x`` -> ``Simulation.x``.

    Both functions live in ``rfx/sparams/dispatch.py`` as module-level
    ``def``s, so they must set ``__qualname__`` at the module foot for that
    composition-time rewrite to see them --
    ``tests/unit/autodiff/test_design_mask_removed.py
    ::test_no_public_simulation_method_leaks_a_mixin_class_name`` is the
    general gate; this is the specific one.
    """
    assert Simulation.compute_s_matrix.__qualname__ == (
        "Simulation.compute_s_matrix")
    assert Simulation.s_matrix_lane.__qualname__ == "Simulation.s_matrix_lane"


def test_a_bad_keyword_names_the_delegate_not_the_dispatcher():
    """``**kwargs`` swallows an unknown keyword here; the delegate rejects it.

    Documented behaviour, not an accident: the delegate's name is the more
    useful one because it reports WHICH LANE was chosen as well as which
    argument was wrong. Pinned so a future signature change on the dispatcher
    (which would move the error to ``Simulation.compute_s_matrix()``) is a
    conscious edit.
    """
    with pytest.raises(TypeError) as exc:
        _waveguide_sim().compute_s_matrix(num_periodz=1)
    msg = str(exc.value)
    assert msg.startswith("Simulation.compute_waveguide_s_matrix()"), msg
    assert "num_periodz" in msg


def test_lane_is_consumed_and_never_forwarded_to_the_delegate():
    """No delegate has a ``lane`` parameter; forwarding it would TypeError.

    Sentinel rather than a solve: the stub records what it was handed.
    """
    seen = {}

    def spy(self, **kwargs):
        seen.update(kwargs)
        return "ok"

    sim = _waveguide_sim()
    Simulation.compute_waveguide_s_matrix  # binding exists before patching
    try:
        original = Simulation.compute_waveguide_s_matrix
        Simulation.compute_waveguide_s_matrix = spy
        assert sim.compute_s_matrix(
            lane="compute_waveguide_s_matrix", num_periods=1) == "ok"
    finally:
        Simulation.compute_waveguide_s_matrix = original
    assert seen == {"num_periods": 1}


# ---------------------------------------------------------------------------
# Equivalence. Real FDTD, bit identity, no tolerance.
# ---------------------------------------------------------------------------

def _assert_same_result(via_dispatcher, direct, s_field: str, leg: str):
    """Bit identity on the S array, plus equal freqs and port names.

    ``np.array_equal``, never a tolerance: the dispatcher forwards the same
    kwargs to the same function on the same host in the same process, so
    anything short of identity means the two paths are not the same
    computation and the difference has to be explained, not absorbed.
    """
    a = np.asarray(getattr(via_dispatcher, s_field))
    b = np.asarray(getattr(direct, s_field))
    assert a.dtype == b.dtype, f"{leg}: dtype {a.dtype} != {b.dtype}"
    assert a.shape == b.shape, f"{leg}: shape {a.shape} != {b.shape}"
    assert np.array_equal(a, b, equal_nan=True), (
        f"{leg}: compute_s_matrix() and the direct call disagree on "
        f"{s_field}. They forward identical kwargs to the same function, so "
        "this is a dispatcher defect, not a tolerance question. "
        f"max |diff| = {np.nanmax(np.abs(a - b))}"
    )
    fa = np.asarray(via_dispatcher.freqs)
    fb = np.asarray(direct.freqs)
    assert np.array_equal(fa, fb), f"{leg}: freqs differ"
    assert via_dispatcher.port_names == direct.port_names, (
        f"{leg}: port_names differ -- {via_dispatcher.port_names} vs "
        f"{direct.port_names}")


@pytest.mark.slow_physics
def test_waveguide_lane_matches_the_direct_call():
    """Default waveguide lane, ``num_periods=1`` (the #980 lock's leg)."""
    kw = dict(normalize=False, num_periods=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via = _waveguide_sim().compute_s_matrix(**kw)
        direct = _waveguide_sim().compute_waveguide_s_matrix(**kw)
    _assert_same_result(via, direct, "s_params", "waveguide")


@pytest.mark.slow_physics
def test_msl_lane_matches_the_direct_call():
    kw = dict(freqs=_MSL_FREQS, num_periods=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via = _msl_sim().compute_s_matrix(**kw)
        direct = _msl_sim().compute_msl_s_matrix(**kw)
    _assert_same_result(via, direct, "S", "msl")


@pytest.mark.slow_physics
def test_coaxial_two_port_lane_matches_the_direct_call():
    """The ambiguous row, driven through its ``lane=`` selector.

    Also the check that ``lane=`` is consumed rather than forwarded on a real
    solve, not just against the sentinel above.
    """
    kw = dict(n_steps=400, freqs=_COAX_BAND)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via = _coax_sim().compute_s_matrix(
            lane="compute_coaxial_two_port", **kw)
        direct = _coax_sim().compute_coaxial_two_port(**kw)
    _assert_same_result(via, direct, "s_params", "coaxial_two_port")


@pytest.mark.slow_physics
def test_coax_msl_transition_lane_matches_the_direct_call():
    from tests._coax_msl_instrument_fixture import instrument_kwargs

    kw = instrument_kwargs(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via = _coax_msl_sim().compute_s_matrix(**kw)
        direct = _coax_msl_sim().compute_coax_msl_transition(**kw)
    _assert_same_result(via, direct, "s_params", "coax_msl_transition")


# ---------------------------------------------------------------------------
# The docstring table is the user-facing copy of the dispatch logic. Keep the
# two from drifting the way the repo-map copy did (see
# .claude/rules/rfx-feature-discovery.md).
# ---------------------------------------------------------------------------

def test_every_selectable_lane_appears_in_the_docstring_table():
    from rfx.sparams import dispatch

    doc = Simulation.compute_s_matrix.__doc__
    assert doc is not None
    for lane in dispatch._SELECTABLE_LANES:
        name = lane if lane != dispatch.RUN_LANE else "compute_s_params=True"
        assert name in doc, (
            f"{name!r} is selectable but absent from compute_s_matrix's "
            "documented dispatch table")


def test_every_documented_delegate_is_a_real_simulation_method():
    from rfx.sparams import dispatch

    for lane in dispatch._SELECTABLE_LANES:
        if lane == dispatch.RUN_LANE:
            assert re.match(r"^run\(", lane)
            assert callable(Simulation.run)
            continue
        assert callable(getattr(Simulation, lane)), (
            f"{lane!r} is selectable but is not a Simulation method")
