"""Shared Pydantic configuration for every public contract."""

import json

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Rejects unknown keys and type coercion; instances are immutable."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def load_strict[M: StrictModel](model: type[M], data: object) -> M:
    """Validate JSON-like data (parsed TOML/JSON, table rows) against a strict model.

    Validation goes through JSON so that arrays become tuples and strings become enums
    exactly as they would when reading a file, while other coercions stay forbidden.
    Non-finite floats are rejected because JSON cannot represent them.
    """
    try:
        text = json.dumps(data, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"{model.__name__}: non-finite number in input") from error
    return model.model_validate_json(text)
