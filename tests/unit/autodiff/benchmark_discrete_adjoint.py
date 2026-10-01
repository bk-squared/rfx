"""CPU-only #1424 measurement of the unit gate fixture; no performance assertion.

Run each option in a fresh process, with PYTHONPATH pointing at this worktree:
  python -m tests.unit.autodiff.benchmark_discrete_adjoint --gradient adjoint
"""
import argparse
import json
import platform
import resource
import statistics
import time

import jax
import jax.numpy as jnp

from tests.unit.autodiff.test_discrete_adjoint import fixture, objective_fn, STEPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient", choices=("autodiff", "adjoint"), required=True)
    parser.add_argument("--objective", choices=("probe_dft", "ntff"), default="probe_dft")
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("This measurement is CPU only")
    sim, eps = fixture(objective=args.objective)
    sigma = jnp.full_like(eps, 0.02)
    loss = objective_fn(sim, args.objective, gradient=args.gradient, checkpoint_segments=4)
    function = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    start = time.perf_counter()
    executable = function.lower(eps, sigma).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(executable(eps, sigma))
    seconds = []
    for _ in range(7):
        start = time.perf_counter()
        jax.block_until_ready(executable(eps, sigma))
        seconds.append(time.perf_counter() - start)
    memory = executable.memory_analysis()
    stats = None if memory is None else {
        k: getattr(memory, k) for k in (
            "argument_size_in_bytes", "output_size_in_bytes",
            "alias_size_in_bytes", "temp_size_in_bytes")}
    print(json.dumps(dict(
        gradient=args.gradient, objective=args.objective, jax=jax.__version__,
        platform=platform.platform(), machine=platform.machine(), backend="cpu",
        grid_shape=list(sim._build_grid().shape), design_shape=list(eps.shape),
        n_steps=STEPS, checkpoint_segments=4, precision="float32", sigma=0.02,
        compile_seconds=compile_seconds, seconds=seconds,
        median_seconds=statistics.median(seconds), compiled_memory_analysis=stats,
        process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        process_peak_rss_units="bytes" if platform.system() == "Darwin" else "KiB",
        process_peak_rss_scope="whole process including imports, compilation and execution",
        interpretation="lead fills"), indent=2))


if __name__ == "__main__":
    main()
