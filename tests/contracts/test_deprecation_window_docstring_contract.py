"""Docstrings and runtime warnings must state the same contract.

The source default now names a meaning (#1373), not a future version window.
Removed periodic/PEC/time-gating APIs are covered by test_api_1448.py (#1448).
"""

import inspect
import re
import warnings

from rfx import Simulation



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


def test_add_source_docstring_and_warning_state_the_same_meaning():
    doc = _normalize(Simulation.add_source.__doc__)
    sim = Simulation(freq_max=1e9, domain=(0.01, 0.01, 0.01))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        sim.add_source(position=(0.005, 0.005, 0.005))
    warning = _normalize(str(next(
        w.message for w in captured if issubclass(w.category, DeprecationWarning))))
    pattern = r"None\)?(?: now)? means (.+?) on every path"
    doc_meaning = re.findall(pattern, doc)
    warning_meaning = re.findall(pattern, warning)
    assert len(doc_meaning) == len(warning_meaning) == 1
    assert doc_meaning == warning_meaning
    assert doc_meaning == ["'current' (E += Cb*I/dV, I is a current moment in A·m)"]
