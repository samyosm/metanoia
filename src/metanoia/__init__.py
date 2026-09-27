"""Metanoia core inference.

Top-level import stays lightweight: PyTorch is imported only by the modules that need it.
Core must never import ``metanoia_research`` or ``metanoia_analysis``.
"""

__version__ = "0.1.0"
