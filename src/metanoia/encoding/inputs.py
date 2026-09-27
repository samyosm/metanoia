"""Rendering a request into encoder text. Part of the predictor's identity."""

from metanoia_contracts import DecisionRequest

RENDERING_REVISION = "question-context-observations/1"


def render_input(request: DecisionRequest) -> str:
    """Question first, so truncation removes the end of the context rather than the question."""
    parts = [f"question: {request.question}", f"context: {request.context}"]
    if request.observations:
        items = "; ".join(
            f"{key} = {request.observations[key]}" for key in sorted(request.observations)
        )
        parts.append(f"observations: {items}")
    return "\n".join(parts)


def render_candidate_joint(candidate_text: str, rendered_input: str) -> str:
    """Candidate first, so truncation of a long context never removes the candidate."""
    return f"answer: {candidate_text}\n{rendered_input}"
