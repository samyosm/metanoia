"""Versioned trace events: detached summaries of what the predictor actually computed.

A trace records execution; it is not a proof (SCIENTIFIC_FOUNDATIONS §10).
"""

from typing import Literal

from pydantic import Field

from metanoia_contracts.base import StrictModel

TRACE_VERSION = 1

TraceValue = str | int | float | bool | None | list[float] | list[str]


class TraceEvent(StrictModel):
    """One operation. ``inputs``/``outputs`` are node or state identities."""

    trace_version: Literal[1] = 1
    seq: int = Field(ge=0)
    op: str = Field(min_length=1)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    details: dict[str, TraceValue] = Field(default_factory=dict)
