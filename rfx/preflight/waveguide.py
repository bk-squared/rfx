"""Waveguide-port preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 3. The ports_waveguide family: the realized-aperture
and cutoff checks (#150 / #737 / #738), the reference-plane sanity check
(P2.8), and the three post-v1.8 S-parameter setup audits with the builder
they share. Everything here was relocated byte for byte out of
``rfx/api/_preflight.py`` -- same text, same order, same indentation, same
docstrings, nothing renamed, reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first. That extension was chosen by a CALL CENSUS rather than by counting
unwitnessed codes, because this family is the one where the two disagree:
``_check_waveguide_port_evanescent_declared_geometry`` emits through the same
shared emitter as the uniform lane, so every slug it can raise was already
"witnessed" while the body itself had never been entered by any fixture. The
lock module's own docstring carries the measurement.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

Two things live here. This first half is the module block -- the three
module-level waveguide leaves. ``rfx.api._preflight`` re-exports all three,
which is not optional for any of them:

* ``_waveguide_skipped_note`` is called by BARE NAME from the two setup
  audits that consume ``_waveguide_far_geometry``. They stay on
  ``_PreflightMixin`` until the check bodies follow, so until then the name
  has to resolve as a global of the facade.
* ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` and ``resolve_waveguide_port_freqs`` are
  read as bare module globals by ``preflight_sparameters``, which STAYS in
  the facade permanently -- it is the per-calculator routing entry point, not
  a family check. ``tests/unit/preflight/test_waveguide_setup_audits.py``
  also imports the constant from ``rfx.api._preflight`` and pins it against
  ``compute_waveguide_s_matrix``'s live signature, and
  ``rfx/sparams/waveguide.py`` imports the resolver from there too.

``rfx/sparams/waveguide.py``'s import is deliberately NOT repointed at this
module, which is where leg 1 went the other way. Leg 1 repointed
``rfx/sparams/msl.py`` and ``rfx/sparams/mixed.py`` at ``rfx.preflight.msl``
because ``msl_probe_clearance_for_port`` is monkeypatched and
``tests/unit/ports/test_msl_clearance_diagnostic.py`` asserts the patched
function is observed from the preflight side AND the S-matrix side, which
needs both readers resolving through ONE lookup target. An AST sweep over
``tests/ rfx/ validation/ scripts/ examples/``, resolving aliases and
``importlib`` forms and string-form patch targets, finds NO patch site on any
of the three names here. With no patch to keep consistent there is no second
lookup target to collapse, so the facade re-export -- whose object identity
``tests/locks/test_preflight_split_snapshot.py`` pins per name -- is left as
the path, and a pure code-motion leg touches one fewer file.
"""

from __future__ import annotations

import jax.numpy as jnp


def _waveguide_skipped_note(skipped: list) -> str:
    """The trailing sentence naming ports whose launch direction was unreadable.

    Shared by both audits that consume :meth:`_waveguide_far_geometry`, so the
    two cannot describe the same skip differently.
    """
    if not skipped:
        return ""
    return (" Ports skipped because their launch direction could not "
            f"be read: {', '.join(skipped)}.")


# ``compute_waveguide_s_matrix``'s own ``num_periods`` default, mirrored here
# so ``preflight_sparameters(calculator="waveguide")`` audits the record a user
# gets when they pass nothing. Pinned to the live signature by
# tests/unit/preflight/test_waveguide_setup_audits.py.
WAVEGUIDE_DEFAULT_NUM_PERIODS = 20.0


def resolve_waveguide_port_freqs(sim, entry):
    """The measured frequency grid of one waveguide port entry.

    ONE definition, two users: ``compute_waveguide_s_matrix`` resolves the
    band with it and so does ``preflight_sparameters(calculator="waveguide")``,
    so the setup audits can never be reading a different band than the run.
    """
    if entry.freqs is not None:
        return entry.freqs
    return jnp.linspace(sim._freq_max / 10, sim._freq_max, entry.n_freqs)
