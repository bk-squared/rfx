# Codex Spec 6E2: Field Decay Convergence Criterion

## Goal
Replace fixed `num_periods` simulation length with an energy-decay-based
stopping criterion, so simulations run exactly as long as needed for accurate
S-parameters.

## Background
Meep uses `stop_when_fields_decayed(dT, component, point, decay_by)`:
- Monitors field amplitude at a point
- Stops when |field|² has decayed to `decay_by` fraction of peak
- Typical: `decay_by=1e-3` (standard), `1e-9` (high-accuracy)

rfx currently uses `grid.num_timesteps(num_periods=N)` which is a fixed
count unrelated to whether fields have actually settled.

## Deliverable

### 1. `run_until_decay()` in `rfx/simulation.py`

```python
def run_until_decay(
    grid,
    materials,
    *,
    decay_by: float = 1e-3,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50000,
    monitor_component: str = "ez",
    monitor_position: tuple[float, float, float] | None = None,
    # ... same kwargs as run(): sources, probes, boundary, etc.
) -> SimResult:
    """Run simulation until field energy decays to `decay_by` of peak.

    Parameters
    ----------
    decay_by : float
        Stop when |field|² < decay_by * peak|field|².
    check_interval : int
        Check decay every N steps.
    min_steps : int
        Always run at least this many steps.
    max_steps : int
        Hard upper limit on steps.
    monitor_component : str
        Field component to monitor ("ez", "hy", etc.).
    monitor_position : tuple or None
        Physical position to monitor. If None, use center of domain.
    """
```

Implementation approach:
- Cannot use `jax.lax.scan` (needs dynamic termination)
- Use `jax.lax.while_loop` with a carry that includes:
  - FDTD state, step counter, peak amplitude, current amplitude, converged flag
- OR use a Python loop with JIT-compiled step function (simpler, slight overhead)
- The Python loop approach is recommended for initial implementation since
  the overhead of Python loop control is negligible vs FDTD step cost

### 2. High-level API: `Simulation.run(until_decay=1e-3)`

Add `until_decay: float | None = None` to `Simulation.run()`.
When provided, overrides `n_steps` and uses the decay criterion.

```python
result = sim.run(until_decay=1e-3)  # auto-determines simulation length
result = sim.run(n_steps=500)       # existing fixed-length behavior
```

### 3. Tests in `tests/test_decay_convergence.py`

**Test 1: `test_decay_stops_after_pulse_exits`**
- Small PEC cavity with Gaussian pulse
- `run_until_decay(decay_by=1e-2)` should stop when pulse rings down
- Assert: final |field|² / peak < decay_by
- Assert: stopped before max_steps

**Test 2: `test_decay_produces_better_dft_than_fixed_short`**
- Compare DFT quality: `run(n_steps=100)` vs `run_until_decay(decay_by=1e-3)`
- Decay run should have smoother frequency response (less truncation ripple)
- Measure: variance of |S11| across frequency band

**Test 3: `test_decay_min_max_steps_honored`**
- `min_steps=200`: even if decayed early, run at least 200 steps
- `max_steps=50`: should stop at 50 even if not decayed
- Assert step counts

## Constraints
- Do NOT modify the existing `run()` compiled scan path — add a separate function
- The existing `run(n_steps=N)` must continue to work unchanged
- Each test < 60 seconds
- Python loop is acceptable for the decay path (JIT step function + Python control)

## Verification
Run: `pytest -xvs tests/test_decay_convergence.py`
All 3 tests must pass. Full suite must not regress.
