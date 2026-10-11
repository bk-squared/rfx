"""The realized model (S1): one declaration + one grid -> one object every path reads.

``realized_view`` exposes a read-only NumPy snapshot of the realized cell volumes.
``conductors`` holds the PEC edges, sheets, lossy-sheet operators and wires.
"""

from rfx.model.view import RealizedView, realized_view

__all__ = ["RealizedView", "realized_view"]
