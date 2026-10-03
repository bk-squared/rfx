"""Message numbers do not depend on how numpy prints a scalar (numpy 1 vs 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _structured_snapshot import message_numbers  # noqa: E402


def test_numpy_scalar_repr_does_not_add_numbers():
    numpy2 = "gap np.float64(0.005) m is np.int64(3) cells"
    numpy1 = "gap 0.005 m is 3 cells"
    assert message_numbers(numpy2) == message_numbers(numpy1) == [0.005, 3.0]
