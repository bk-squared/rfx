"""Contract: a deprecation's docstring and its runtime warning name the same versions.

Issue #954. ``Simulation.add_source``'s docstring kept announcing the old
``amplitude_kind`` window ("From 1.8 required; from 1.9 the default") after the
window moved to required-in-1.9 / default-in-2.0 (``980a4a86``), while the
``DeprecationWarning`` raised a few lines below in the same method carried the
new one. Two statements of one schedule drifted apart across a release because
nothing compared them.

This file compares them. For each deprecation that states its schedule in BOTH
the docstring and the warning literal of the same function, the version numbers
must match. It is deliberately mechanical: it does not model deprecation policy,
does not read ``pyproject.toml`` or ``CHANGELOG.md``, and covers only the
deprecations that carry a version in both places.

Not covered, and why (checked at the time of writing):

* ``pec_faces=`` — the warning says "removed in rfx v2.0" but no docstring in
  ``rfx/api/__init__.py`` states a version for it, so there is no pair.
* ``compute_coaxial_s_matrix`` (``rfx/api/_sparams.py``) and
  ``minimize_s11_at_freq`` (``rfx/optimize_objectives.py``) — issue #954 lists
  them as living in ``rfx/api/__init__.py``; they do not.
"""

import inspect
import re

from rfx import Simulation

_REQUIRED_IN = r"required in (\d+\.\d+)"
_DEFAULT_IN = r"default in (\d+\.\d+)"
_REMOVED_IN = r"removed in (?:rfx )?v?(\d+\.\d+)"


def _normalize(text: str) -> str:
    """Flatten RST markup and line wrapping so a sentence split across source
    lines reads the same as the single-line warning literal."""
    return re.sub(r"\s+", " ", text.replace("``", "").replace("`", ""))


def _docstring_and_body(func) -> tuple[str, str]:
    """Return (raw docstring, function source with the docstring removed).

    Splitting on the raw ``__doc__`` keeps the two statements apart: whatever
    is left of the source holds the ``warnings.warn`` literal.
    """
    src = inspect.getsource(func)
    doc = func.__doc__
    assert doc, f"{func.__qualname__} has no docstring"
    assert doc in src, (
        f"{func.__qualname__}: docstring not found verbatim in its source; "
        "the split below would be meaningless"
    )
    return doc, src.replace(doc, "", 1)


def _versions(text: str, pattern: str) -> list[str]:
    return sorted(set(re.findall(pattern, _normalize(text))))


def test_add_source_docstring_and_warning_state_the_same_window():
    doc, body = _docstring_and_body(Simulation.add_source)

    doc_required = _versions(doc, _REQUIRED_IN)
    warn_required = _versions(body, _REQUIRED_IN)
    doc_default = _versions(doc, _DEFAULT_IN)
    warn_default = _versions(body, _DEFAULT_IN)

    # Phrasing has to stay parallel for the comparison to mean anything: both
    # statements say "required in X" and "default in Y".
    assert len(doc_required) == 1, (
        "add_source docstring must state the amplitude_kind window as "
        f"'required in X'; found {doc_required}"
    )
    assert len(warn_required) == 1, (
        "add_source DeprecationWarning must state 'required in X'; "
        f"found {warn_required}"
    )
    assert len(doc_default) == 1, (
        "add_source docstring must state the amplitude_kind window as "
        f"'default in Y'; found {doc_default}"
    )
    assert len(warn_default) == 1, (
        "add_source DeprecationWarning must state 'default in Y'; "
        f"found {warn_default}"
    )

    assert doc_required == warn_required, (
        "add_source docstring and DeprecationWarning disagree on when "
        f"amplitude_kind becomes required: docstring {doc_required[0]}, "
        f"warning {warn_required[0]} (issue #954)"
    )
    assert doc_default == warn_default, (
        "add_source docstring and DeprecationWarning disagree on when "
        f"'current' becomes the default: docstring {doc_default[0]}, "
        f"warning {warn_default[0]} (issue #954)"
    )


def test_set_periodic_axes_docstring_and_warning_state_the_same_removal():
    doc, body = _docstring_and_body(Simulation.set_periodic_axes)

    doc_removed = _versions(doc, _REMOVED_IN)
    warn_removed = _versions(body, _REMOVED_IN)

    assert len(doc_removed) == 1, (
        "set_periodic_axes docstring must state 'removed in vX.Y'; "
        f"found {doc_removed}"
    )
    assert len(warn_removed) == 1, (
        "set_periodic_axes DeprecationWarning must state 'removed in vX.Y'; "
        f"found {warn_removed}"
    )
    assert doc_removed == warn_removed, (
        "set_periodic_axes docstring and DeprecationWarning disagree on the "
        f"removal version: docstring {doc_removed[0]}, warning "
        f"{warn_removed[0]} (issue #954)"
    )
