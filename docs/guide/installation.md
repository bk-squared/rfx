# Installation

## Basic Install

```bash
pip install rfx-fdtd
```

Requires Python 3.10+ and installs JAX, NumPy, SciPy, matplotlib, and h5py.

The package name on PyPI is `rfx-fdtd`, while the Python import remains:

```python
import rfx
```

## GPU Support

rfx uses JAX for computation. To enable GPU acceleration:

```bash
# Install JAX with CUDA support (NVIDIA GPU)
pip install --upgrade "jax[cuda12]"

# Verify GPU is detected
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)]
```

See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for detailed GPU setup including CUDA and cuDNN requirements.

## Development Install

```bash
git clone https://github.com/BK3536/rfx.git
cd rfx
pip install -e ".[dev]"
```

The `dev` extra includes pytest, pytest-xdist, and ruff:

```bash
pytest -x -q  # Run all tests
```

## Optional Dependencies

| Package | Purpose |
|---------|---------|
| `optax` | Alternative optimizers for inverse design |
| `meep` | Cross-validation tests only |
| `openEMS` | Cross-validation tests only |

## System Requirements

- **CPU**: Any modern x86_64 (ARM via JAX experimental)
- **GPU**: NVIDIA with CUDA 12+ and cuDNN 8.9+ (optional, 10-50x speedup)
- **RAM**: 4GB minimum, 16GB+ recommended for large 3D grids
- **Python**: 3.10, 3.11, 3.12
