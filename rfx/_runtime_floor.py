"""Runtime contract for source/PYTHONPATH installs as well as pip (#1429)."""

import re

MIN_PYTHON = (3, 11)
MIN_JAX = "0.10.2"
# Install-time cap only (#2116): from jaxlib 0.11.2 the CPU time step runs 2.3-3x slower; import does not refuse newer jax.
JAX_BELOW = "0.11.2"


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
    """Raise an actionable ImportError when any runtime component is too old."""
    if (tuple(python_version_info[:2]) >= MIN_PYTHON
            and _version_key(jax_version) >= _version_key(MIN_JAX)
            and _version_key(jaxlib_version) >= _version_key(MIN_JAX)):
        return
    python_found = ".".join(map(str, python_version_info[:3]))
    python_required = ".".join(map(str, MIN_PYTHON))
    raise ImportError(
        f"rfx runtime floor (#1429): found Python {python_found}, "
        f"jax {jax_version}, jaxlib {jaxlib_version}; required Python >= "
        f"{python_required}, jax >= {MIN_JAX}, jaxlib >= {MIN_JAX}. "
        f"Create a Python {python_required} venv and install "
        f"jax=={MIN_JAX} jaxlib=={MIN_JAX}, as scripts/vessl_gpu_suite.yaml "
        "does; or use rfx 1.8.x."
    )
