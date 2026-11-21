"""Report the host witness's block shape with the round-3 rule."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rfx.measurement import matrix_pencil as mp

mp._RESIDUE_BLOCK_TARGET_BYTES = 17 * 16 * (4 + 2)
size = mp._residue_block_size(313, 4, 2)
print(dict(length=313, rank=4, channels=2, target=mp._RESIDUE_BLOCK_TARGET_BYTES,
           Mb=size, blocks=(313 + size - 1) // size, last_block=313 % size or size))
