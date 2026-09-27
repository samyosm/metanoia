"""Independent scoring of saved artifacts.

Imports only ``metanoia_contracts``, NumPy, SciPy, PyArrow and (for figures)
Matplotlib. It never loads checkpoints, invokes inference, or imports PyTorch.
"""

__version__ = "0.1.0"
ANALYSIS_REVISION = f"metanoia_analysis-{__version__}"
