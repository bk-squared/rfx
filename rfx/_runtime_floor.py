"""Runtime contract for source/PYTHONPATH installs as well as pip (#1429)."""

import re
import warnings

from rfx.errors import ConfigurationWarning

MIN_PYTHON = (3, 11)
MIN_JAX = "0.10.2"
# Install cap and runtime warning (#2116): from jaxlib 0.11.2 the CPU time step runs 2.3-3x slower; import does not refuse newer jax.
JAX_TESTED_BELOW = "0.11.2"
JAX_BELOW = JAX_TESTED_BELOW


class UntestedJAXWarning(ConfigurationWarning):
    """The installed JAX version is outside the tested range."""


def _version_key(version: str) -> tuple[tuple[int, ...], int]:
    """Order JAX versions; the one comparison for MIN_JAX and JAX_BELOW (#2133).

    The release segment compares numerically at any length (0.12.0.1 above 0.12.0,
    0.10.2.0 equal to 0.10.2); dev/alpha/beta/rc sort before their release. Local
    build and post-release suffixes meet the corresponding release. Unknown version
    formats fail closed.
    """
    match = re.fullmatch(
        r"(\d+(?:\.\d+)*)((?:(?:\.dev|a|b|rc)\d*)?(?:\.post\d+)?(?:\+[-\w.]+)?)",
        version,
    )
    if match is None:
        return ((-1,), 0)
    release, suffix = match.groups()
    parts = [int(part) for part in release.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    prerelease = suffix.startswith((".dev", "a", "b", "rc"))
    return (tuple(parts), int(not prerelease))


def check(python_version_info, jax_version: str, jaxlib_version: str) -> None:
    """Reject runtimes below the floor and warn above the tested JAX range."""
    if (tuple(python_version_info[:2]) >= MIN_PYTHON
            and _version_key(jax_version) >= _version_key(MIN_JAX)
            and _version_key(jaxlib_version) >= _version_key(MIN_JAX)):
        if any(_version_key(v) >= _version_key(JAX_TESTED_BELOW)
               for v in (jax_version, jaxlib_version)):
            warnings.warn(
                f'rfx found jax {jax_version}, jaxlib {jaxlib_version}; '
                f'tested range: >= {MIN_JAX}, < {JAX_TESTED_BELOW}. '
                f'Install with: pip install "jax<{JAX_TESTED_BELOW}" '
                f'"jaxlib<{JAX_TESTED_BELOW}".',
                UntestedJAXWarning, stacklevel=2,
            )
        return
    python_found = ".".join(map(str, python_version_info[:3]))
    python_required = ".".join(map(str, MIN_PYTHON))
    raise ImportError(
        f"rfx runtime floor: found Python {python_found}, "
        f"jax {jax_version}, jaxlib {jaxlib_version}; required Python >= "
        f"{python_required}, jax >= {MIN_JAX}, jaxlib >= {MIN_JAX}. "
        f'Create a Python >= {python_required} venv and run '
        f'pip install "jax>={MIN_JAX}" "jaxlib>={MIN_JAX}"; '
        "or use rfx 1.8.x."
    )
