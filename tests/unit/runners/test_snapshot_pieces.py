"""The scan layout of a snapshot-recording run (#1258), without running one.

``run()`` records a frame every ``interval`` steps by scanning the steps in
pieces laid out by ``rfx.snapshots.plan_snapshot_pieces``; ``record()`` in
``run()`` executes exactly those pieces and keeps exactly their
``frame_rows``. Two measured failures shaped the layout, and each is an
invariant here:

* a step run as a one-step scan is inlined by XLA and moved the last bit of
  Ex/Ey on a PEC box, so no piece may be one step long unless the whole
  segment is;
* joining that step with a WHOLE neighbouring block made a scan that output
  every step's frame for ``interval + 2`` steps -- 767 MiB peak against
  443 MiB on a 72^3 full-field run at interval 200, about 15 GiB projected
  for a 200^3 field at interval 500 -- so the piece that outputs per-step
  frames must stay at three steps or fewer, whatever the interval.

Pure Python: 28 125 layouts in well under a second.
"""
import pytest

from rfx.snapshots import plan_snapshot_pieces


@pytest.mark.parametrize("m", range(2, 17))
def test_piece_layout_invariants(m):
    for lo in range(0, 3 * m):
        for n in range(0, 6 * m + 3):
            pieces = plan_snapshot_pieces(n, lo, m)
            where = f"interval={m} lo={lo} n={n}: {pieces}"
            # The pieces tile the segment in step order.
            assert [p.start for p in pieces] == [
                sum(q.length for q in pieces[:k]) for k in range(len(pieces))
            ], where
            assert sum(p.length for p in pieces) == n, where
            frames = []
            for p in pieces:
                assert p.kind in ("plain", "blocks", "rec"), where
                # No step runs as a one-step scan unless the segment is one.
                assert p.length >= 2 or len(pieces) == 1, where
                if p.kind == "blocks":
                    assert (lo + p.start) % m == 0 and p.length % m == 0, where
                    assert p.frame_rows == tuple(range(m - 1, p.length, m)), where
                elif p.kind == "plain":
                    assert p.frame_rows in ((), (p.length - 1,)), where
                else:
                    # A per-step-frame piece holds at most three frames.
                    assert 2 <= p.length <= 3, where
                frames += [lo + p.start + r + 1 for r in p.frame_rows]
            # Every frame lands on a multiple, and every multiple has one.
            assert frames == [t for t in range(lo + 1, lo + n + 1)
                              if t % m == 0], where


def test_large_interval_keeps_the_per_step_piece_small():
    for m in (50, 200, 500):
        for lo, n in ((0, 3 * m + 1), (m - 1, 5 * m + 1), (1, 2 * m + 2)):
            rec = [p.length for p in plan_snapshot_pieces(n, lo, m)
                   if p.kind == "rec"]
            assert all(L <= 3 for L in rec), (m, lo, n, rec)
