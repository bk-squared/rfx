"""PR A float64 model invariant, rfx#2613; DESIGN A1-A5 and A2b-A2f.

R1 reads realized_operands at boundaries.tfsf.admit_setup; R2 its pre-widening
mesh; R3 init_cpml's return; R4 recorded_scan inputs and closed-body constants;
R5 pole_component_weights returns, labeled by their consuming _realize_poles.
Every scene declares 200 steps (A2f); capture stops before the first step.
P1 applies everywhere, P2 only at R4/R5. Equation outputs/literals are not read
(A2b). P2 is computed from bits independently in float64_readpoints.py.

FLOAT32_UNSEEN_2613 records the unreached FIX sites and the reached conformal
face areas consumed before the declared reads. U-smooth's True declaration
selects full node tensors, already float64, not the old Kottke cast paths.
UPML source coefficients (materials.py:692,694) fall with A1's refusal.

The pending + unseen ceiling only falls, like scripts/ci/check_file_size_ratchet.py:
never increase the integer to admit a new violation. Remove fixed observed
names; transfer an unseen name to pending only with evidence, keeping the sum
at or below the ceiling. All named exceptions must retain their census reason.
"""
import jax
import pytest

from tests._x64_compat import enable_x64
from tests.contracts.float64_pending import (
    FLOAT32_BY_CONTRACT, FLOAT32_PENDING_2613, FLOAT32_UNSEEN_2613, PENDING_CEILING,
)
from tests.contracts.float64_readpoints import SCENES, capture


def test_float64_model_invariant():
    allowed = FLOAT32_BY_CONTRACT | FLOAT32_PENDING_2613
    assert not FLOAT32_BY_CONTRACT.keys() & FLOAT32_PENDING_2613.keys()
    assert all(allowed.values()) and all(FLOAT32_UNSEEN_2613.values())
    assert len(FLOAT32_PENDING_2613) + len(FLOAT32_UNSEEN_2613) <= PENDING_CEILING
    seen, unexpected = set(), []
    with enable_x64():
        for label in SCENES:
            if label.endswith('tfsf'):
                # A2d: split once; this scene already contains only one object.
                for _ in range(2):
                    with pytest.raises(ValueError, match='update.materials.eps_r prevents an invariant periodic replacement'):
                        capture(label)
                continue
            rows = capture(label)
            for point, name, dtype, shape, p1, p2 in rows:
                if p1 or p2:
                    seen.add(name)
                    if name not in allowed:
                        unexpected.append((label, point, name, dtype, shape, 'P1' if p1 else 'P2'))
            jax.clear_caches()
    assert not unexpected, f'new float64 violations: {unexpected}'
    assert not allowed.keys() - seen, f'fixed: remove it: {sorted(allowed.keys() - seen)}'
