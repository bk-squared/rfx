"""W5.5 contract tests: forward() docstring completeness + curated __all__.

Fast, dependency-free introspection guards so the public surface cannot drift
silently:

1. Every keyword argument of ``Simulation.forward`` is documented in its
   docstring ``Parameters`` section.
2. ``rfx.__all__`` is a curated subset (materially smaller than the full flat
   export surface) and every listed name resolves on the package.
3. ``from rfx import *`` binds exactly ``__all__`` and every name is importable.
"""

import inspect
import re

import rfx
from rfx import Simulation


def _parameters_section(docstring: str) -> str:
    """Return the text of the NumPy-style ``Parameters`` section only.

    Slices from the ``Parameters`` header to the next section header
    (``Returns``/``Raises``/``Examples``/...) so a kwarg name that only
    appears in ``Returns`` prose does not count as documented.
    """
    assert docstring, "Simulation.forward has no docstring"
    lines = docstring.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "Parameters" and i + 1 < len(lines) and set(
            lines[i + 1].strip()
        ) == {"-"}:
            start = i + 2
            break
    assert start is not None, "forward() docstring has no 'Parameters' section"

    end = len(lines)
    for j in range(start, len(lines) - 1):
        underline = lines[j + 1].strip()
        if lines[j].strip() and underline and set(underline) == {"-"} and len(
            underline
        ) >= 3:
            end = j
            break
    return "\n".join(lines[start:end])


def test_forward_every_kwarg_documented():
    sig = inspect.signature(Simulation.forward)
    kwargs = [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert kwargs, "forward() exposes no keyword arguments — introspection broke"

    params_text = _parameters_section(inspect.getdoc(Simulation.forward))

    missing = []
    for name in kwargs:
        # Match a NumPy-style parameter entry: name at start of a line,
        # followed by ' :' or end-of-line (type annotation optional).
        pattern = rf"(?m)^\s*{re.escape(name)}\s*(:|$)"
        if not re.search(pattern, params_text):
            missing.append(name)

    assert not missing, (
        "forward() kwargs missing from the docstring Parameters section: "
        f"{missing}"
    )


def test_all_is_curated_subset():
    assert hasattr(rfx, "__all__"), "rfx defines no __all__"
    names = rfx.__all__
    assert isinstance(names, list)
    assert len(names) == len(set(names)), "rfx.__all__ contains duplicates"
    # Curated: materially smaller than the full flat surface (318 public names
    # on the package as of the AD-certificate/diagnostics/pareto surface;
    # __all__=204, ratio 0.64 — tighter than the 181/~245=0.74 this gate was
    # written against). Ceiling re-specced 200 -> 210 for that deliberate
    # 23-name expansion; kept tight on purpose so the NEXT surface expansion
    # trips this gate again and must re-justify itself.
    assert len(names) < 210, f"rfx.__all__ too large to be curated: {len(names)}"
    missing = [n for n in names if not hasattr(rfx, n)]
    assert not missing, f"rfx.__all__ lists names not on the package: {missing}"


def test_all_excludes_per_step_internals():
    # The curated surface must not re-export per-step kernel internals.
    forbidden_prefixes = ("init_", "update_", "inject_", "apply_waveguide_port_")
    leaked = [
        n
        for n in rfx.__all__
        if n.startswith(forbidden_prefixes)
        or n in {"thomas_solve", "adi_step_2d", "adi_step_3d"}
    ]
    assert not leaked, f"per-step internals leaked into rfx.__all__: {leaked}"


def test_star_import_binds_all_and_is_importable():
    ns: dict = {}
    exec("from rfx import *", ns)
    star_names = {n for n in ns if not n.startswith("__")}
    assert star_names == set(rfx.__all__), (
        "from rfx import * did not bind exactly __all__; "
        f"symmetric_difference={star_names ^ set(rfx.__all__)}"
    )
    for name in rfx.__all__:
        assert ns[name] is getattr(rfx, name)
