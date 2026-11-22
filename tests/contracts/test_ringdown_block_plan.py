"""Real scan boundaries obey the shared tall-skinny block contract on CPU."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import matrix_pencil as pencil
from rfx.measurement import ringdown_blocked as blocked


@pytest.mark.parametrize('fit', ['device', 'host'])
def test_balanced_scan_plan(monkeypatch, fit):
    """Inspect the boundaries consumed by all passes, including short windows.

    Device scans are intercepted before computation; host row construction
    records its actual ranges. Both must consume the plan without deriving
    boundaries again. Sweep target counts above and below the width floor.
    """
    width = 5
    for length in (3, 5, 6, 10, 11, 16, 21, 31):
        for target_rows in (1, 4, 5, 6, 11, 100):
            with monkeypatch.context() as patch, blocked.jax.default_device(blocked.jax.devices('cpu')[0]):
                seen = []
                if fit == 'device':
                    patch.setattr(blocked, '_BLOCK_TARGET_BYTES', target_rows * width * 8)
                    plan = blocked._block_plan(length, width, jnp.complex64)

                    def scan(body, initial, windows):
                        seen.append(tuple(map(tuple, np.asarray(windows).tolist())))
                        return initial, None

                    patch.setattr(blocked.jax.lax, 'scan', scan)
                    y = jnp.ones((length, 1), jnp.complex64)
                    s = jnp.array([-.1 + 1j, -.2 + 2j], jnp.complex64)
                    mask = jnp.ones(2)
                    blocks = blocked._factor(y, s, .01, mask, plan)
                    blocked._products(y, s, .01, mask, *blocks[-2:], plan)
                else:
                    patch.setattr(pencil, '_RESIDUE_BLOCK_TARGET_BYTES', target_rows * width * 16)
                    plan = pencil._residue_block_plan(length, 4, 1)
                    original = pencil.np.arange
                    ranges = []

                    def arange(start, stop, **kwargs):
                        ranges.append((start, stop))
                        return original(start, stop, **kwargs)

                    patch.setattr(pencil.np, 'arange', arange)
                    pencil._fit_residues(np.ones((length, 1)),
                                         np.array([-.1 + j*1j for j in range(4)]), .01)
                    seen = [tuple(ranges[:len(plan)]), tuple(ranges[len(plan):])]
                    memory = pencil._residue_memory_bytes(length, 4, 1)
                    assert memory == 1.71 * 16 * 5 * (max(b-a for a, b in plan) * width + width**2)
                assert len(seen) == (3 if fit == 'device' else 2)
                for windows in seen:
                    assert windows == plan, 'scan must consume the shared plan'
                    sizes = [stop-start for start, stop in windows]
                    assert max(sizes)-min(sizes) <= 1, 'short remainder restored'
                    assert min(sizes) >= min(width, length), 'block shorter than width'
                    assert windows[0][0] == 0 and windows[-1][1] == length
                    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))
                    assert len(windows) == min((length+target_rows-1)//target_rows,
                                               max(1, length//width))
