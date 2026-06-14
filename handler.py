"""HuggingFace Inference Endpoint custom handler (HF entrypoint).

Deployed by pointing an HF Inference Endpoint at a repo containing this file and
a `requirements.txt`. Request payload:

    {"inputs": "...", "direction": "en-ng", "parameters": {...}}

`model_id` (e.g. "meyabase/en-ng-translation") is accepted in place of `direction`
for backward compatibility.
"""
from __future__ import annotations

from typing import Any

from translation.core import Translator


class EndpointHandler:
    def __init__(self, path: str = "") -> None:
        self.translator = Translator()

    def __call__(self, data: dict) -> Any:
        inputs = data.get("inputs")
        if inputs is None:
            raise ValueError("Request must include 'inputs'.")
        parameters = data.get("parameters") or {}
        direction = data.get("direction")
        model_id = data.get("model_id")
        return self.translator.translate(
            inputs, direction=direction, model_id=model_id, **parameters
        )
