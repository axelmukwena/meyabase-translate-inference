"""Configuration: translation directions and environment handling."""
from __future__ import annotations

import os

# Friendly direction key -> HuggingFace Hub model id.
DIRECTIONS: dict[str, str] = {
    "en-ng": "meyabase/en-ng-translation",
    "ng-en": "meyabase/ng-en-translation",
}

_MODEL_IDS = set(DIRECTIONS.values())


def resolve_model_id(*, direction: str | None = None, model_id: str | None = None) -> str:
    """Resolve a request to a Hub model id.

    Accepts either a friendly ``direction`` ("en-ng") or a full ``model_id``
    ("meyabase/en-ng-translation") for backward compatibility with existing callers.
    """
    if direction is not None:
        try:
            return DIRECTIONS[direction]
        except KeyError:
            raise ValueError(
                f"Unknown direction {direction!r}. Valid directions: {sorted(DIRECTIONS)}"
            )
    if model_id is not None:
        if model_id in _MODEL_IDS:
            return model_id
        raise ValueError(
            f"Unknown model_id {model_id!r}. Valid model ids: {sorted(_MODEL_IDS)}"
        )
    raise ValueError(
        f"Must provide a direction or model_id. Valid directions: {sorted(DIRECTIONS)}"
    )


def get_hf_token() -> str | None:
    """Return the HuggingFace token from the environment, if set."""
    return os.environ.get("TF_TOKEN")
