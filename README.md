# rfx

**A differentiable 3-D electromagnetic simulator for RF and microwave engineering, built in JAX.**

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/bk-squared/rfx/blob/main/LICENSE)
[![Tests](https://github.com/bk-squared/rfx/actions/workflows/pr-tests.yml/badge.svg)](https://github.com/bk-squared/rfx/actions)
[![PyPI](https://img.shields.io/pypi/v/rfx-fdtd)](https://pypi.org/project/rfx-fdtd/)

See waves propagate, measure a component's response, and differentiate a design objective.
Start with the [visual manual and examples](https://remilab.ai/rfx/).
The website identifies its source version; select the release matching your installation.

| You want to… | Start here |
|---|---|
| See results computed with the current code | [Showcase](https://remilab.ai/rfx/showcase/) · [visual archive](https://remilab.ai/rfx/gallery/) |
| Run a first model | [First run](https://remilab.ai/rfx/guide/first-run/) · [runnable examples](https://github.com/bk-squared/rfx/blob/main/examples/README.md) |
| Look up an argument or result | [API reference](https://remilab.ai/rfx/api/) |
| Use rfx with a coding assistant | [Machine-readable documentation](https://remilab.ai/rfx/llms.txt) |
| Check whether a result is supported | [Support matrix](https://github.com/bk-squared/rfx/blob/main/docs/guides/support_matrix.md) · [S-parameter limits](https://github.com/bk-squared/rfx/blob/main/docs/guides/sparameter_support_matrix.md) · [known limitations](https://github.com/bk-squared/rfx/blob/main/docs/guides/known_limitations.md) |

## Release status: 2.0 release candidate

`2.0.0rc1` is the first release candidate of rfx 2.0. What 2.0 changes for a user:

- **An input a solver path does not implement is refused, never dropped.** If `run()`,
  `forward()` or an S-parameter calculator routes your model to a path that does not
  solve a declared input (a conformal wall, subpixel smoothing, a lumped element, a port,
  a dispersive material, a solver or precision request), it raises before the first time
  step and names the input. It does not run a different model silently.
- **Validation cases are judged against a stated accuracy bar** on a converged mesh:
  magnitudes within 2 dB, resonances and cutoffs within 1 %, electrical length within 1 %
  of the reference. This is the bar the project tests against, not a guarantee for every
  model. What is supported, and within which limits, is in the
  [support matrix](https://github.com/bk-squared/rfx/blob/main/docs/guides/support_matrix.md); what is currently wrong is in
  [known limitations](https://github.com/bk-squared/rfx/blob/main/docs/guides/known_limitations.md).
- **Python 3.11 or newer**, with JAX and jaxlib 0.10.2 or newer. Python 3.10 users stay on 1.8.x.
- Multi-device (distributed) execution is not part of 2.0's supported set; it is planned
  for 2.1.

## Install

```bash
pip install rfx-fdtd            # latest stable release
pip install --pre rfx-fdtd      # 2.0 release candidate
```

For GPU execution, use a compatible JAX/CUDA installation:

```bash
pip install "jax[cuda12]" rfx-fdtd
```

For the development version, clone this repository and run `pip install -e ".[dev]"`.
The [installation guide](https://remilab.ai/rfx/guide/installation/) covers optional packages.

## First run

A pulse in a small closed PEC box checks your installation. It records one field
probe; it does not measure a device's S-parameters, resonance accuracy, or Q.
This code is generated from [hello_world.py](https://github.com/bk-squared/rfx/blob/main/examples/quickstart/hello_world.py),
the same runnable example tested by CI.

<!-- rfx-hello-world:start -->

```python
from rfx import GaussianPulse, Simulation

sim = Simulation(
    freq_max=10e9,
    domain=(0.02, 0.02, 0.02),
    dx=2e-3,
    boundary="pec",
)
sim.add_source(
    (0.01, 0.01, 0.01),
    "ez",
    waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
    amplitude_kind="current",
)
sim.add_probe((0.014, 0.01, 0.01), "ez")

preflight = sim.preflight()
print(preflight.format())
preflight.raise_for_failure()
n_steps = 120
result = sim.run(n_steps=n_steps, compute_s_params=False)
print(result.time_series.shape)
```

<!-- rfx-hello-world:end -->

All lengths are metres and frequencies are hertz. Read the preflight findings.
Continue with [building and measuring a model](https://remilab.ai/rfx/guide/quickstart/).

## Accuracy and development

Support depends on the structure, mesh, boundary and observable. A successful run
or an agreeing automatic/finite-difference gradient is not a convergence study.
The [accuracy guide](https://remilab.ai/rfx/guide/validation/) explains the checks;
[benchmarks](https://remilab.ai/rfx/guide/benchmarks/) link results to their scope.

Contributor workflow and repository navigation live in
[Working on rfx](https://github.com/bk-squared/rfx/blob/main/docs/agent/working-on-rfx.mdx) and the [repo map](https://github.com/bk-squared/rfx/blob/main/docs/agent/repo-map.mdx).
Release changes are in [CHANGELOG.md](https://github.com/bk-squared/rfx/blob/main/CHANGELOG.md).

## Citation

```bibtex
@article{kim_rfx_tmtt_2026,
  author = {Byungkwan Kim},
  title = {rfx: An End-to-End Differentiable 3-D FDTD Simulator for RF and Microwave Engineering},
  journal = {IEEE Transactions on Microwave Theory and Techniques},
  year = {2026},
  pages = {1--15},
  doi = {10.1109/TMTT.2026.3729863}
}
```

The software itself:

```bibtex
@software{kim_rfx_2026,
  author = {Byungkwan Kim},
  title = {rfx: JAX-based differentiable 3D FDTD simulator for RF engineering},
  institution = {REMI Lab, Chungnam National University},
  year = {2026},
  url = {https://github.com/bk-squared/rfx}
}
```

MIT License — see [LICENSE](https://github.com/bk-squared/rfx/blob/main/LICENSE). Developed at the
[Radar & ElectroMagnetic Intelligence Laboratory](https://remilab.cnu.ac.kr).
