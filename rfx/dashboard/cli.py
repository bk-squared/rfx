"""Console entry point for the rfx Streamlit dashboard.

Exposes ``rfx-dashboard`` (see ``[project.scripts]`` in ``pyproject.toml``),
a thin launcher that runs the Streamlit app shipped at
``rfx/dashboard/app.py`` via ``streamlit run``.

Usage::

    rfx-dashboard                       # launch on the default port (8501)
    rfx-dashboard --server.port 8502    # forward extra flags to streamlit
    rfx-dashboard --help                # show this usage (no streamlit needed)

The dashboard is an experimental convenience GUI; install it with::

    pip install "rfx-fdtd[dashboard]"
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_USAGE = """\
usage: rfx-dashboard [streamlit options]

Launch the rfx Streamlit dashboard — an experimental browser GUI for building
and running FDTD simulations and viewing S-parameters, Smith charts, field
slices, and Touchstone exports without writing code.

Any extra arguments are forwarded verbatim to `streamlit run`, e.g.:

    rfx-dashboard --server.port 8502
    rfx-dashboard --server.address 0.0.0.0 --server.headless true

Requires the dashboard extra:

    pip install "rfx-fdtd[dashboard]"
"""

_MISSING_STREAMLIT_MSG = (
    'Streamlit is not installed. Install the dashboard extra:  '
    'pip install "rfx-fdtd[dashboard]"'
)


def _app_path() -> Path:
    """Absolute path to the Streamlit app, resolved from the package location."""
    return Path(__file__).parent / "app.py"


def main(argv: list[str] | None = None) -> int:
    """Launch the rfx Streamlit dashboard.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.
            Any arguments are forwarded to ``streamlit run`` unchanged, except
            ``-h``/``--help`` which print local usage without importing
            streamlit.

    Returns:
        Process exit code (streamlit's exit code, or non-zero on error).
    """
    if argv is None:
        argv = sys.argv[1:]

    if "-h" in argv or "--help" in argv:
        sys.stdout.write(_USAGE)
        return 0

    if importlib.util.find_spec("streamlit") is None:
        sys.stderr.write(_MISSING_STREAMLIT_MSG + "\n")
        return 1

    app_path = _app_path()
    cmd = ["streamlit", "run", str(app_path), *argv]

    try:
        completed = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        # streamlit importable but console script absent: run via the module.
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *argv]
        completed = subprocess.run(cmd, check=False)

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
