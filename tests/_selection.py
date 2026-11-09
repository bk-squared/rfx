"""Session selection records shared by census tests."""
from collections import Counter
from pathlib import Path

import pytest

DESELECTED_PATHS = pytest.StashKey[Counter[Path]]()


def deselected_count(config, path) -> int:
    return config.stash.get(DESELECTED_PATHS, Counter())[Path(path).resolve()]
