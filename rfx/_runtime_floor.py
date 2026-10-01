"""Runtime contract for source/PYTHONPATH installs as well as pip (#1429)."""

import re

MIN_PYTHON = (3, 11)
MIN_JAX = "0.10.2"


def _version_key(version: str) -> tuple[int, int, int, int]:
    """Compare JAX release triplets, with dev/alpha/beta/rc before final.

    Local build and post-release suffixes meet the corresponding release floor.
    Unknown version formats fail closed.
    """
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)((?:(?:\.dev|a|b|rc)\d*)?(?:\.post\d+)?(?:\+[-\w.]+)?)",
        version,
    )
    if match is None:
        return (-1, -1, -1, -1)
    major, minor, patch, suffix = match.groups()
    prerelease = suffix.startswith((".dev", "a", "b", "rc"))
    return (int(major), int(minor), int(patch), int(not prerelease))


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
