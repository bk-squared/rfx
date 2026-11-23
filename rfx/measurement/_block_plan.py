"""Balanced row boundaries shared by device and host tall-skinny fits."""


def block_plan(length, width, target_bytes, itemsize):
    """Cap the byte target's block count so every real block has width rows.

    The target requests ceil(length / max(1, target_bytes // row_bytes))
    blocks. Cap that count at floor(length / width), with at least one
    block, then distribute real rows with sizes differing by at most one.
    A window shorter than width is one block. Boundaries exclude padding.
    """
    target_rows = max(1, int(target_bytes) // (itemsize * width))
    count = min((length + target_rows - 1) // target_rows, max(1, length // width))
    size, extra = divmod(length, count)
    starts = tuple(i * size + min(i, extra) for i in range(count + 1))
    return tuple(zip(starts[:-1], starts[1:]))
