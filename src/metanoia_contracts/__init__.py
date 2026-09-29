"""Public, language-neutral contracts shared by the model, the runner, and the analysis.

This package imports only the standard library and Pydantic. It must never import
``metanoia``, ``metanoia_research`` or ``metanoia_analysis``.
"""

from metanoia_contracts.requests import (
    CONTRACT_VERSION,
    Budget,
    Candidate,
    DecisionRequest,
    TraceMode,
)
from metanoia_contracts.results import (
    CalibrationInfo,
    CalibrationStatus,
    DecisionResult,
    ResourceUsage,
    Termination,
)
from metanoia_contracts.traces import TraceEvent
from metanoia_contracts.updates import DecisionUpdate, NodeView

__version__ = "0.1.0"

__all__ = [
    "CONTRACT_VERSION",
    "Budget",
    "CalibrationInfo",
    "CalibrationStatus",
    "Candidate",
    "DecisionRequest",
    "DecisionResult",
    "DecisionUpdate",
    "NodeView",
    "ResourceUsage",
    "Termination",
    "TraceEvent",
    "TraceMode",
    "__version__",
]
