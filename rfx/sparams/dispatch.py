"""One entry point that picks the right S-matrix lane (issue #980 Phase 1).

``Simulation`` ships seven S-parameter calculators plus a run-lane, and each
one is fenced to a single port family by its own precondition block. Issue
#980 section 1 records what that costs: an agent (or a new user) calls
``sim.run(compute_s_params=True)`` on a simulation whose ports are waveguide
or coaxial -- where that lane computes nothing at all -- or guesses a method
name such as ``compute_s_params()`` that does not exist.

:func:`compute_s_matrix` reads the registrations, picks the one calculator
that covers them, and forwards to it. :func:`s_matrix_lane` answers the same
question and returns the chosen method's NAME without running any FDTD, so
the table is inspectable and testable on a registration-only ``Simulation``.

Design rules this module holds itself to
----------------------------------------

*No invented defaults.* Every keyword argument is forwarded to the delegate
verbatim; the delegate owns its own defaults, validation and error messages.
One consequence is user-visible and deliberate: a misspelled keyword is
swallowed by ``**kwargs`` here and reported by the delegate, so the
``TypeError`` reads ``Simulation.compute_waveguide_s_matrix() got an
unexpected keyword argument 'foo'`` rather than naming this dispatcher. That
is the more useful message -- it says which lane was chosen as well as which
argument was wrong -- and it is pinned by
``tests/unit/sparams/test_compute_s_matrix_dispatch.py``.

*No silent lane guess.* Where the registration genuinely does not determine a
lane, this module raises and names the candidates rather than picking one.
That is the whole point of the dispatcher: the failure mode #980 describes is
a wrong lane running quietly, not a missing convenience.

The one ambiguous registration: a single coaxial port
-----------------------------------------------------

``compute_coaxial_line_reflection`` and ``compute_coaxial_two_port`` have
IDENTICAL registration footprints. Both require exactly one
``add_coaxial_port()`` (``len(self._coaxial_ports) != 1`` raises in both),
both reject every other port family, and both build their own fixture
geometry from that single registration -- the two-port method mirrors the
one-port end into a through-line internally, it does not consume a second
registered port. Nothing in the registrations distinguishes "one-port
reflection against a calibration termination" from "two-port through line",
so this module will not choose between them. Pass ``lane=`` to say which:

    sim.compute_s_matrix(lane="compute_coaxial_two_port", n_steps=3000)

``lane`` is consumed here and never forwarded. It is accepted on any row
(where it must agree with the registration-determined lane, or ``ValueError``
is raised), so calling code can be explicit everywhere rather than only on
the coaxial row.

Import contract, inherited from ``rfx.api._sparams``: import ONLY
``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy, never the
``rfx.api`` package itself, so ``rfx/api/__init__.py`` stays the sole
composition point and the import graph stays acyclic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rfx.api import Simulation  # noqa: F401


# The lumped/wire family has no ``compute_*`` method at all: its S-parameters
# come back as ``Result.s_params`` from the run lane. ``s_matrix_lane()``
# returns this string for that row so the table stays total; it is not a
# method name and ``compute_s_matrix()`` refuses to call ``run()`` for the
# caller (different return type, and ``n_steps`` has no default there).
RUN_LANE = "run(compute_s_params=True)"

# The two lanes a single ``add_coaxial_port()`` registration can mean. See the
# module docstring: their registration footprints are identical.
COAXIAL_ONE_PORT_LANES = (
    "compute_coaxial_line_reflection",
    "compute_coaxial_two_port",
)

# Every lane name ``s_matrix_lane()`` can return, for validating ``lane=``.
# ``compute_coaxial_s_matrix`` is deliberately absent: it is the DEPRECATED
# single-plane lane (its own docstring records measured, non-physical
# ``|S11|>1`` on a lossless short), so the dispatcher never selects it and
# ``lane=`` cannot ask for it either. Call it directly if you need it.
_SELECTABLE_LANES = (
    "compute_waveguide_s_matrix",
    "compute_msl_s_matrix",
    "compute_coaxial_line_reflection",
    "compute_coaxial_two_port",
    "compute_coax_msl_transition",
    "compute_mixed_s_matrix",
    RUN_LANE,
)


def _port_census(self) -> dict:
    """Count every registration that decides an S-matrix lane.

    ``add_port(impedance=0)`` / ``add_source()`` share ``self._ports`` with
    real lumped/wire ports (``rfx/api/__init__.py`` stores a soft source as a
    ``_PortEntry`` with ``impedance=0.0``), but they are not ports for
    S-parameter purposes and ``compute_mixed_s_matrix`` rejects them outright
    -- they would fire in every drive run. Counted separately so the error
    messages can say which of the two is present.
    """
    lumped_wire = [pe for pe in self._ports if pe.impedance != 0.0]
    bare_sources = [pe for pe in self._ports if pe.impedance == 0.0]
    return {
        "waveguide": len(self._waveguide_ports),
        "msl": len(self._msl_ports),
        "coaxial": len(self._coaxial_ports),
        "lumped_wire": len(lumped_wire),
        "bare_source": len(bare_sources),
        "floquet": len(self._floquet_ports),
        "tfsf": 1 if self._tfsf is not None else 0,
    }


_CENSUS_LABELS = {
    "waveguide": "add_waveguide_port",
    "msl": "add_msl_port",
    "coaxial": "add_coaxial_port",
    "lumped_wire": "add_port",
    "bare_source": "add_source / add_port(impedance=0)",
    "floquet": "add_floquet_port",
    "tfsf": "add_tfsf_source",
}


def _census_summary(census: dict) -> str:
    """``"add_waveguide_port x2, add_msl_port x1"`` -- registered only."""
    present = [
        f"{_CENSUS_LABELS[key]} x{count}"
        for key, count in census.items()
        if count
    ]
    return ", ".join(present) if present else "nothing"


def _no_ports_message() -> str:
    return (
        "compute_s_matrix() needs at least one registered port. Register "
        "one of:\n"
        "  sim.add_waveguide_port(...)  -> compute_waveguide_s_matrix()\n"
        "  sim.add_msl_port(...)        -> compute_msl_s_matrix()\n"
        "  sim.add_coaxial_port(...)    -> compute_coaxial_line_reflection() "
        "or compute_coaxial_two_port()\n"
        "  sim.add_port(...)            -> res = sim.run(n_steps=..., "
        "compute_s_params=True); res.s_params\n"
        "add_source(), add_tfsf_source() and add_floquet_port() are not "
        "S-matrix port families and have no lane."
    )


def _resolve_lane(self) -> str:
    """The lane the registrations determine, or raise saying why they don't.

    Split out of :func:`s_matrix_lane` so the ``lane=`` override is applied
    in exactly one place.
    """
    c = _port_census(self)

    def _only(*keys) -> bool:
        """True when every family OUTSIDE ``keys`` is unregistered.

        Every row below is written as "this family is present AND nothing
        else is", rather than as a list of the families it excludes, so a
        NEW port family added to the census cannot silently fall into an
        existing row -- it lands in the catch-all and raises.
        """
        return not any(v for k, v in c.items() if k not in keys)

    if not any(c.values()):
        raise ValueError(_no_ports_message())

    # --- the six determinate rows ---------------------------------------
    if c["waveguide"] and _only("waveguide"):
        return "compute_waveguide_s_matrix"

    if c["msl"] and _only("msl"):
        return "compute_msl_s_matrix"

    if c["coaxial"] == 1 and c["msl"] == 1 and _only("coaxial", "msl"):
        return "compute_coax_msl_transition"

    if c["lumped_wire"] and c["msl"] and _only("lumped_wire", "msl"):
        return "compute_mixed_s_matrix"

    if c["lumped_wire"] and _only("lumped_wire"):
        return RUN_LANE

    if c["coaxial"] == 1 and _only("coaxial"):
        raise NotImplementedError(
            "one add_coaxial_port() registration is shared by TWO validated "
            "lanes and the registrations do not distinguish them:\n"
            "  compute_coaxial_line_reflection(...) -- one-port reflection "
            "against a calibration termination (short / open / matched), "
            "returns CoaxialLineReflectionResult\n"
            "  compute_coaxial_two_port(...) -- two-drive through-line "
            "2-port, returns CoaxialTwoPortResult\n"
            "Both build their own fixture geometry from that single port "
            "(each raises if len(sim._coaxial_ports) != 1), so "
            "compute_s_matrix() will not guess. Say which:\n"
            "  sim.compute_s_matrix(lane='compute_coaxial_two_port', ...)\n"
            "or call the method directly."
        )

    # --- everything else -------------------------------------------------
    raise NotImplementedError(_no_lane_message(c))


def _no_lane_message(c: dict) -> str:
    """Name the registrations, then the nearest single-family method."""
    lines = [
        "no S-matrix lane covers this combination of registrations: "
        f"{_census_summary(c)}."
    ]

    if c["tfsf"]:
        lines.append(
            "add_tfsf_source() is a plane-wave source, not a port, and every "
            "S-matrix calculator rejects it; there is no TFSF S-matrix lane."
        )
    if c["floquet"]:
        lines.append(
            "add_floquet_port() has no S-matrix calculator on Simulation at "
            "all; every compute_* method above rejects it."
        )
    if c["bare_source"]:
        lines.append(
            "add_source() / add_port(impedance=0) register a bare soft "
            "source on self._ports. It is not excite-gated, so it would fire "
            "in every drive run and contaminate the single-drive "
            "S-parameter contract; compute_mixed_s_matrix() rejects it "
            "explicitly and the single-family methods reject any self._ports "
            "entry. Remove it, or use a separate simulation."
        )
    if c["coaxial"] > 1:
        lines.append(
            f"{c['coaxial']} coaxial ports are registered. Every validated "
            "coaxial lane is built from EXACTLY ONE add_coaxial_port() "
            "(compute_coaxial_line_reflection, compute_coaxial_two_port and "
            "compute_coax_msl_transition each raise otherwise). The only "
            "method that accepts several is compute_coaxial_s_matrix(), "
            "which is DEPRECATED and measured non-physical (|S11|>1 on a "
            "lossless short); this dispatcher never selects it."
        )

    singles = []
    if c["waveguide"]:
        singles.append(
            f"{c['waveguide']} waveguide port(s) -> "
            "compute_waveguide_s_matrix() (waveguide ports only)"
        )
    if c["msl"]:
        singles.append(
            f"{c['msl']} MSL port(s) -> compute_msl_s_matrix() (MSL ports "
            "only), or compute_mixed_s_matrix() with add_port() lumped/wire "
            "ports, or compute_coax_msl_transition() with exactly one "
            "coaxial port"
        )
    if c["coaxial"] == 1:
        singles.append(
            "1 coaxial port -> compute_coaxial_line_reflection() or "
            "compute_coaxial_two_port() (coaxial only), or "
            "compute_coax_msl_transition() with exactly one MSL port"
        )
    if c["lumped_wire"]:
        singles.append(
            f"{c['lumped_wire']} lumped/wire port(s) -> "
            "res = sim.run(n_steps=..., compute_s_params=True); "
            "res.s_params, or compute_mixed_s_matrix() with MSL ports"
        )
    if singles:
        lines.append(
            "Registered families and the method each one has on its own:\n  "
            + "\n  ".join(singles)
            + "\nUse a separate Simulation per family."
        )
    return "\n".join(lines)


def s_matrix_lane(self, *, lane: str | None = None) -> str:
    """Which S-matrix lane this simulation's registrations select.

    Runs no FDTD and touches no grid: it reads the port registries only, so
    it is cheap to call on a half-built ``Simulation`` and it is what the
    dispatch table's tests assert against.

    Parameters
    ----------
    lane : str or None
        Optional explicit lane name, one of the ``compute_*`` names below or
        ``"run(compute_s_params=True)"``. Required on the single-coaxial-port
        row, where two lanes share one registration footprint (see the module
        docstring). On every other row it is checked against the lane the
        registrations determine and raises ``ValueError`` if it disagrees --
        it can select a lane, never override the physics fence.

    Returns
    -------
    str
        The delegate's method name, e.g. ``"compute_waveguide_s_matrix"``, or
        the literal ``"run(compute_s_params=True)"`` for the lumped/wire
        family, which has no ``compute_*`` method.

    Raises
    ------
    ValueError
        No ports registered at all (the message lists the ``add_*_port``
        builders), or a ``lane=`` that contradicts the registrations.
    NotImplementedError
        A single coaxial port (ambiguous -- pass ``lane=``), or any
        combination of families no calculator covers. The message names the
        registered families with their counts.

    Examples
    --------
    >>> sim.add_waveguide_port(...); sim.add_waveguide_port(...)
    >>> sim.s_matrix_lane()
    'compute_waveguide_s_matrix'
    """
    if lane is not None and lane not in _SELECTABLE_LANES:
        raise ValueError(
            f"lane={lane!r} is not a lane compute_s_matrix() can select. "
            f"Valid: {', '.join(repr(n) for n in _SELECTABLE_LANES)}. "
            "(compute_coaxial_s_matrix is deprecated and deliberately not "
            "selectable; call it directly if you need it.)"
        )

    if lane in COAXIAL_ONE_PORT_LANES:
        # The ambiguous row: _resolve_lane() raises rather than choose, so
        # the override has to be applied before it, not after. Still fenced
        # -- the registration must actually BE the ambiguous one.
        c = _port_census(self)
        if c["coaxial"] == 1 and not any(
            v for k, v in c.items() if k != "coaxial"
        ):
            return lane

    resolved = _resolve_lane(self)
    if lane is not None and lane != resolved:
        raise ValueError(
            f"lane={lane!r} contradicts the registrations, which select "
            f"{resolved!r} ({_census_summary(_port_census(self))}). "
            "lane= disambiguates where two lanes share one registration "
            "footprint; it does not override a port-family fence."
        )
    return resolved


def compute_s_matrix(self, *, lane: str | None = None, **kwargs):
    """Compute S-parameters on whichever lane this simulation's ports select.

    A single entry point over the per-family calculators. It inspects the
    registrations, chooses exactly one delegate, and forwards ``**kwargs``
    to it UNCHANGED -- no default is invented here, so the delegate's
    signature, defaults, preconditions, warnings and return type are exactly
    what you would get calling it directly.

    Dispatch table
    --------------

    ===================================  ==========================================  ================================
    Registered                           Delegate                                    Returns
    ===================================  ==========================================  ================================
    waveguide ports only                 ``compute_waveguide_s_matrix``              ``WaveguideSMatrixResult``
    MSL ports only                       ``compute_msl_s_matrix``                    ``MSLSMatrixResult``
    1 coaxial port only                  ``compute_coaxial_line_reflection`` **or**   ``CoaxialLineReflectionResult``
                                         ``compute_coaxial_two_port`` -- ambiguous,  or ``CoaxialTwoPortResult``
                                         pass ``lane=``
    1 coaxial + 1 MSL                    ``compute_coax_msl_transition``             ``CoaxMSLTransitionResult``
    lumped/wire (``add_port``) + MSL     ``compute_mixed_s_matrix``                  ``MixedSMatrixResult``
    lumped/wire only                     ``NotImplementedError`` -- the run lane     (``Result.s_params``)
    nothing                              ``ValueError``                              --
    anything else                        ``NotImplementedError``                     --
    ===================================  ==========================================  ================================

    The lumped/wire family deliberately raises rather than calling ``run()``
    for you: that lane returns a ``Result`` (not an S-matrix result type),
    and ``n_steps`` has no default there, so quietly substituting one would
    be inventing exactly the kind of default this dispatcher refuses to
    invent. Run it yourself::

        res = sim.run(n_steps=..., compute_s_params=True)
        res.s_params

    Parameters
    ----------
    lane : str or None
        Explicit lane name (see :meth:`s_matrix_lane`). Consumed here, never
        forwarded. Required for a single coaxial port, where
        ``compute_coaxial_line_reflection`` and ``compute_coaxial_two_port``
        share one registration footprint; optional elsewhere, where it is
        checked against the registrations rather than overriding them.
    **kwargs
        Forwarded verbatim to the chosen delegate.

    Returns
    -------
    Whatever the chosen delegate returns -- see the table.

    Raises
    ------
    ValueError
        No ports registered, or ``lane=`` contradicts the registrations.
        Also raised by the chosen lane itself: ``compute_coax_msl_transition``
        (1 coaxial + 1 MSL) refuses a non-passive extracted S by default and
        is the only lane in the table that does — forward
        ``strict_passivity=False`` through ``**kwargs`` to get the diagnostic
        matrix with a warning instead (issue #838).
    NotImplementedError
        Lumped/wire-only (use the run lane), a single coaxial port with no
        ``lane=``, or a combination of port families no calculator covers.
    TypeError
        An unrecognised keyword argument. ``**kwargs`` swallows it here, so
        the message names the DELEGATE that rejected it --
        ``Simulation.compute_waveguide_s_matrix() got an unexpected keyword
        argument 'foo'`` -- which also tells you which lane was chosen.

    Notes
    -----
    Preflight behaviour is inherited from the chosen lane and is not uniform
    across the table: ``compute_msl_s_matrix`` and ``compute_mixed_s_matrix``
    run preflight automatically, the waveguide and coaxial lanes do not. See
    each delegate's row in
    ``tests/unit/preflight/test_preflight_advisory_emission_contract.py``.

    See Also
    --------
    s_matrix_lane : the same choice, without running anything.
    """
    chosen = self.s_matrix_lane(lane=lane)

    if chosen == RUN_LANE:
        raise NotImplementedError(
            "lumped/wire ports (add_port) have no compute_* S-matrix "
            "method: their S-parameters come back from the run lane.\n"
            "  res = sim.run(n_steps=..., compute_s_params=True)\n"
            "  res.s_params\n"
            "compute_s_matrix() does not call run() for you -- it returns a "
            "Result rather than an S-matrix result type, and its n_steps has "
            "no default that this dispatcher could honestly supply."
        )

    # Dispatched by explicit ``self.<method>(...)`` calls, not getattr: the
    # emission-classification gate
    # (tests/unit/preflight/test_preflight_advisory_emission_contract.py
    # ::test_emission_classification_matches_measured_reachability) walks
    # ``self.foo(...)`` calls through the AST to measure whether an entry
    # point reaches preflight. A getattr indirection would hide this method's
    # real reachability from that gate and let it be classified
    # DIAGNOSTIC_ONLY on a technicality -- the "silently widening what counts
    # as diagnostic-only" failure that gate exists to stop.
    if chosen == "compute_waveguide_s_matrix":
        return self.compute_waveguide_s_matrix(**kwargs)
    if chosen == "compute_msl_s_matrix":
        return self.compute_msl_s_matrix(**kwargs)
    if chosen == "compute_coaxial_line_reflection":
        return self.compute_coaxial_line_reflection(**kwargs)
    if chosen == "compute_coaxial_two_port":
        return self.compute_coaxial_two_port(**kwargs)
    if chosen == "compute_coax_msl_transition":
        return self.compute_coax_msl_transition(**kwargs)
    if chosen == "compute_mixed_s_matrix":
        return self.compute_mixed_s_matrix(**kwargs)

    raise AssertionError(  # pragma: no cover -- guards a typo in the table
        f"s_matrix_lane() returned {chosen!r}, which compute_s_matrix() has "
        "no branch for. The two tables have drifted."
    )


# ---------------------------------------------------------------------------
# ``__qualname__``, matched to the #980 Phase 2 convention.
#
# These two are NEW methods rather than moved ones, but they are bound into
# ``_SparamMixin``'s class body the same way the moved calculators are, and
# ``rfx/api/__init__.py`` rewrites exactly ``_SparamMixin.<name>`` ->
# ``Simulation.<name>`` at class-composition time, SKIPPING any function whose
# qualname does not match that pattern. A module-level ``def`` gets the bare
# name, so without these two lines ``Simulation.compute_s_matrix.__qualname__``
# would read ``compute_s_matrix`` and the class-name-leak gate
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` -- plus this
# dispatcher's own qualname test -- would see an unrewritten name.
# ---------------------------------------------------------------------------
compute_s_matrix.__qualname__ = "_SparamMixin.compute_s_matrix"
s_matrix_lane.__qualname__ = "_SparamMixin.s_matrix_lane"
