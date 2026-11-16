"""Keep cross-test censuses together after pytest-split selects a shard."""

import pytest

WHOLE_FILE_MODULES = ("tests/contracts/test_finding_subjects.py",)
_SNAPSHOTS = pytest.StashKey[dict[str, tuple[pytest.Item, ...]]]()


def keep_whole(items, snapshot):
    """Give the first item's shard the ordered snapshot; remove it elsewhere.

    Other items retain their relative order. Neither input is mutated, and
    removed contains only items that were actually in this shard.
    """
    if not snapshot:
        return list(items), []
    members = set(snapshot)
    if snapshot[0] not in items:
        return ([item for item in items if item not in members],
                [item for item in items if item in members])

    kept = []
    for item in items:
        if item == snapshot[0]:
            kept.extend(snapshot)
        elif item not in members:
            kept.append(item)
    return kept, []


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not getattr(config.option, "splits", None):
        return
    # Root conftest imports this hook before pytest_configure registers the
    # split plugin. Pluggy calls later-registered trylast hooks last, so this
    # snapshot follows -m/--deselect filtering and precedes the split.
    config.stash[_SNAPSHOTS] = {
        module: tuple(item for item in items
                      if item.nodeid.split("::", 1)[0] == module)
        for module in WHOLE_FILE_MODULES
    }


@pytest.hookimpl(hookwrapper=True, specname="pytest_collection_modifyitems")
def pytest_collection_modifyitems_whole_files(config, items):
    outcome = yield
    outcome.get_result()
    if not getattr(config.option, "splits", None):
        return
    for snapshot in config.stash[_SNAPSHOTS].values():
        kept, removed = keep_whole(items, snapshot)
        items[:] = kept
        if removed:
            config.hook.pytest_deselected(items=removed)
