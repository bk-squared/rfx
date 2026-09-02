"""Issue #662 — the ring-down settling witness must ENFORCE its own -40 dB bar.

The witness (``settling_db``) and its threshold were both already written down;
what was missing on two lanes was the comparison between them. Measured on this
worktree BEFORE the fix (``compute_coaxial_two_port``, the
``tests/unit/sparams/test_coax_two_port_fdtd.py`` through-line fixture, JAX_PLATFORMS=cpu):

    n_steps=  400  settling_db=[ -6.84,  -6.93]  warnings emitted: 0
    n_steps=  700  settling_db=[-28.15, -29.46]  warnings emitted: 0
    n_steps= 1500  settling_db=[-43.97, -44.53]  warnings emitted: 0
    n_steps= 3000  settling_db=[-67.26, -68.09]  warnings emitted: 0
    n_steps= 6000  settling_db=[-65.87, -65.59]  warnings emitted: 0

The 400- and 700-step rows violate the bar this method's own result docstring
documents by 33 and 12 dB, and returned a plausible-looking ``s_params`` in
total silence. ``tests/unit/sparams/test_coax_msl_transition.py`` carries an independent
in-repo sighting of the same silence ("a shorter, 1500-step smoke run measured
only -1.0 / -0.2 dB").

Split, per this repo's physics-run discipline:

  * FAST (no FDTD): the warner's own decision logic — fires / stays silent /
    skips NaN / names every violating drive / names the lane's own knob — plus
    a static governance gate that every ``settling_db``-producing lane routes
    through the ONE shared warner. That gate is the durable part: it is what
    makes a future sixth lane fail loudly instead of repeating #662.
  * SLOW (``slow_physics``, real FDTD): the end-to-end firing pair on the
    silent lane, gates pinned from the measurements above.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import numpy as np
import pytest

from rfx.api._sparams import _SETTLING_WITNESS_DB, _warn_if_ringdown_truncated

_SPARAMS_SRC = pathlib.Path(
    __import__("rfx.api._sparams", fromlist=["_sparams"]).__file__
)


def _catch(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if "settling witness" in str(w.message)]


# ---------------------------------------------------------------------------
# FAST — the warner's decision logic
# ---------------------------------------------------------------------------

def test_threshold_constant_is_the_documented_bar():
    """One shared constant, and it is the -40 dB the docstrings quote."""
    assert _SETTLING_WITNESS_DB == -40.0


def test_violating_witness_warns_and_quotes_the_measured_value():
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.1, -55.0]), ("feed", "load"), n_steps=700))
    assert len(hot) == 1, "one aggregate warning per call, not one per drive"
    msg = str(hot[0].message)
    # The measured value, the bar, the field to inspect, and the knob.
    assert "-1.1 dB" in msg, msg
    assert "-40" in msg and "settling_db" in msg, msg
    assert "n_steps=700" in msg, msg


def test_settled_witness_stays_silent():
    """Control: a check that fires on everything is worse than one that fires
    on nothing. A settled record must produce no warning at all."""
    assert _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-67.26, -68.09]), ("port1", "port2"), n_steps=3000)) == []


def test_witness_exactly_at_the_bar_is_not_a_violation():
    """The bar is documented as "above -40 dB"; equality must not fire (an
    off-by-one here would make the control test above fixture-dependent)."""
    assert _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-40.0, -40.0]), ("a", "b"), n_steps=1)) == []
    assert len(_catch(lambda: _warn_if_ringdown_truncated(
        np.array([-39.9, -80.0]), ("a", "b"), n_steps=1))) == 1


def test_all_nan_witness_is_silent_not_a_false_fire():
    """NaN is a DESIGNED state, not a failure: the differentiable lanes leave
    ``settling_db`` NaN because the witness needs a concrete time series. A
    naive ``settling_db > -40`` would evaluate NaN comparisons; this pins that
    the finite mask, not luck, is what keeps those lanes quiet."""
    assert _catch(lambda: _warn_if_ringdown_truncated(
        np.full(2, np.nan), ("port1", "port2"), n_steps=400)) == []


def test_nan_beside_a_violator_does_not_mask_the_violator():
    """The other half of the NaN decision: a partially-concrete array must
    still report its concrete violator (``np.nanmax``-style silence here would
    be a real regression, and a plain ``np.max`` would return NaN and fire
    never)."""
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        np.array([np.nan, -2.0]), ("port1", "port2"), n_steps=400))
    assert len(hot) == 1
    assert "port port2 driven: -2.0 dB" in str(hot[0].message)
    assert "port1" not in str(hot[0].message)


def test_every_violating_drive_is_named_not_only_the_worst():
    """Record length is a per-drive property with a per-drive remedy; naming
    only the worst drive would hide a second one needing the same fix."""
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.0, -30.0, -70.0]), ("p1", "p2", "p3"), n_steps=400))
    assert len(hot) == 1
    msg = str(hot[0].message)
    assert "port p1 driven: -1.0 dB" in msg and "port p2 driven: -30.0 dB" in msg
    assert "p3" not in msg, "a settled drive must not be named"


def test_warning_names_the_knob_the_lane_is_actually_driven_by():
    """One warning shape, two record-length knobs: the waveguide/MSL/mixed
    lanes are driven by ``num_periods``, the coax lanes by ``n_steps``. Naming
    the wrong one makes the remedy un-actionable."""
    by_periods = str(_catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.0]), ("p1",), num_periods=2.0))[0].message)
    assert "num_periods=2" in by_periods and "Increase num_periods" in by_periods
    by_steps = str(_catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.0]), ("p1",), n_steps=400))[0].message)
    assert "n_steps=400" in by_steps and "Increase n_steps" in by_steps


# ---------------------------------------------------------------------------
# FAST — governance: one warner, wired to every producer
# ---------------------------------------------------------------------------

def _functions_producing_settling_db():
    """(name, routes_through_warner) for every function in ``_sparams.py``
    that attaches a ``settling_db=`` to a result object.

    ``_sparams.py`` is the only module that does so (``grep -rn "settling_db="
    rfx/ --include=*.py`` hits nothing else; ``_spec.py`` only declares the
    field). If a lane is ever added elsewhere, widen this scan with it.
    """
    tree = ast.parse(_SPARAMS_SRC.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        produces = any(
            isinstance(sub, ast.Call)
            and any(kw.arg == "settling_db" for kw in sub.keywords)
            for sub in ast.walk(node)
        )
        if not produces:
            continue
        routed = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "_warn_if_ringdown_truncated"
            for sub in ast.walk(node)
        )
        out.append((node.name, routed))
    return out


def test_every_settling_db_producer_routes_through_the_shared_warner():
    """The #662 defect in structural form.

    RED on the unfixed tree: ``compute_coaxial_two_port`` and
    ``compute_coax_msl_transition`` both attach a ``settling_db`` they never
    compare to the bar. A new lane that copies that pattern fails here rather
    than shipping another silent witness.
    """
    producers = _functions_producing_settling_db()
    assert producers, "AST probe found no settling_db producers — it has rotted"
    silent = sorted(name for name, routed in producers if not routed)
    assert not silent, (
        f"lane(s) {silent} attach a settling_db that is never compared to "
        f"{_SETTLING_WITNESS_DB:g} dB — call _warn_if_ringdown_truncated() "
        "there (issue #662)."
    )


def test_the_known_lanes_are_all_covered():
    """Companion to the gate above: pins WHICH lanes carry the witness, so a
    lane silently losing its witness entirely (producer disappears -> the gate
    above passes vacuously for it) is also caught."""
    names = {name for name, _ in _functions_producing_settling_db()}
    assert {
        "compute_waveguide_s_matrix",
        "compute_msl_s_matrix",
        "compute_mixed_s_matrix",
        "compute_coaxial_two_port",
        "compute_coax_msl_transition",
    } <= names, sorted(names)


# ---------------------------------------------------------------------------
# SLOW — end-to-end on the lane that was silent (real FDTD, ~60 s total)
# ---------------------------------------------------------------------------

_BAND = np.array([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def _coax_two_port_sim():
    """The committed through-line fixture from tests/unit/sparams/test_coax_two_port_fdtd.py
    (domain 8x8x60 mm, freq_max 40 GHz)."""
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


@pytest.mark.slow_physics
def test_underrun_coax_two_port_warns_instead_of_returning_it_quietly():
    """Deliberately under-run record: measured settling_db [-6.84, -6.93] dB
    on this fixture at n_steps=400, i.e. 33 dB past the bar. On the unfixed
    tree this returned a finite, ordinary-looking s_params and zero warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=400, freqs=_BAND)

    sd = np.asarray(res.settling_db)
    assert sd.shape == (2,) and np.all(np.isfinite(sd)), sd
    assert np.all(sd > _SETTLING_WITNESS_DB), (
        f"fixture no longer under-run (settling_db={sd}); it cannot witness "
        "the truncation warning any more — shorten the record."
    )
    hot = [w for w in caught if "settling witness" in str(w.message)]
    assert hot, (
        f"settling_db={sd} violates the {_SETTLING_WITNESS_DB:g} dB bar and "
        "nothing warned (issue #662)"
    )
    msg = str(hot[0].message)
    assert "port port1 driven" in msg and "port port2 driven" in msg, msg
    assert "n_steps=400" in msg and "settling_db" in msg, msg


@pytest.mark.slow_physics
def test_settled_coax_two_port_stays_silent():
    """Control on the SAME fixture: measured [-67.26, -68.09] dB at
    n_steps=3000. A settled run must not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=3000, freqs=_BAND)

    sd = np.asarray(res.settling_db)
    assert np.all(sd < -60.0), (
        f"settled control drifted to {sd}; it no longer sits well clear of "
        "the bar, so its silence would stop meaning anything."
    )
    assert not [w for w in caught if "settling witness" in str(w.message)]


@pytest.mark.slow_physics
def test_differentiable_coax_path_leaves_the_witness_nan_and_silent():
    """The NaN path end-to-end: the ``eps_scale`` lane cannot build the
    witness (it would need a concrete time series), leaves settling_db NaN by
    design, and must therefore stay silent even though the SAME 400-step
    record fires the warning on the concrete lane above."""
    import jax.numpy as jnp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=400, freqs=_BAND, eps_scale=jnp.asarray(1.0))

    sd = np.asarray(res.settling_db)
    assert sd.shape == (2,) and np.all(np.isnan(sd)), sd
    assert not [w for w in caught if "settling witness" in str(w.message)]
