"""A file-wide census belongs to exactly one pytest-split shard."""

import pytest

from tests._whole_file_shards import keep_whole


@pytest.mark.parametrize("items,snapshot,kept,removed", [
    (["before", "a", "after"], ["a", "b", "c"],
     ["before", "a", "b", "c", "after"], []),
    (["before", "a", "between", "c", "after"], ["a", "b", "c"],
     ["before", "a", "b", "c", "between", "after"], []),
    (["before", "b", "between", "c", "after"], ["a", "b", "c"],
     ["before", "between", "after"], ["b", "c"]),
    (["before", "after"], ["a", "b", "c"], ["before", "after"], []),
    (["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"], []),
    ([], ["a", "b", "c"], [], []),
    (["before", "after"], [], ["before", "after"], []),
])
def test_keep_whole(items, snapshot, kept, removed):
    original_items, original_snapshot = list(items), list(snapshot)
    assert keep_whole(items, snapshot) == (kept, removed)
    assert items == original_items
    assert snapshot == original_snapshot
