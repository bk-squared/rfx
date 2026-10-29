"""Lazy imports for optional plotting backends."""


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'matplotlib is required; pip install "rfx-fdtd[visualization]"'
        ) from exc
    return plt
