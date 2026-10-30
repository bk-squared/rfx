"""Compare dielectric inserts with a Pareto front and an epsilon constraint.

Run from the repository root with PYTHONPATH=.: python
examples/inverse_design/pareto_front_sweep.py

The material sweep follows the dielectric-insert sweep in the public sweeps
page, using the smaller PEC box from quickstart/hello_world.py. The two
objectives are peak probe field (maximize) and relative permittivity (minimize,
as a simple material-cost proxy). This short transient illustrates selection;
it is not a converged device design or a frequency-domain measurement.
"""

from __future__ import annotations

from pathlib import Path
import time

import jax
import numpy as np

from examples.quickstart.hello_world import build_simulation
from rfx import Box, parametric_sweep, select_epsilon_constrained


def _make_sim(eps_r):
    sim = build_simulation()
    sim.add_material("insert", eps_r=eps_r)
    sim.add(Box((0.006, 0.006, 0.006), (0.012, 0.014, 0.014)), material="insert")
    return sim


def main() -> None:
    started = time.perf_counter()
    with jax.default_device(jax.devices("cpu")[0]):
        sweep = parametric_sweep(
            _make_sim, "eps_r", [2.0, 3.0, 4.0], n_steps=120,
            run_kwargs={"compute_s_params": False},
        )
    peak = sweep.peak_field()
    front = sweep.pareto_front(
        {"peak_field": peak, "eps_r": sweep.param_values}, minimize=(False, True),
    )
    objectives = np.column_stack([peak, sweep.param_values])
    selected = select_epsilon_constrained(
        objectives, primary_index=0, epsilons={1: 3.0}, minimize=(False, True),
    )
    output = Path(__file__).with_name("pareto_front.json")
    output.write_text(front.to_json() + "\n", encoding="utf-8")
    print("Pareto front indices:", front.front_indices)
    print("Epsilon-constrained choice:", selected)
    print("Choice eps_r:", sweep.param_values[selected])
    print("Peak probe fields:", peak.tolist())
    print("JSON artifact:", output)
    print(f"CPU wall time: {time.perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
