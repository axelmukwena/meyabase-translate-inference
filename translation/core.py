"""Core translation logic shared by every entrypoint."""
from __future__ import annotations

from typing import Any

from translation.config import get_hf_token, resolve_model_id


def extract_translations(prediction: Any) -> Any:
    """Pull ``translation_text`` out of a pipeline prediction.

    Returns a single string when there is one result, else a list of strings.
    """
    if isinstance(prediction, dict):
        return prediction.get("translation_text", "")
    texts = [p.get("translation_text", "") for p in prediction]
    return texts[0] if len(texts) == 1 else texts


def _load_pipeline(model_id: str, device: int) -> Any:
    """Build a HF translation pipeline for ``model_id`` on ``device``.

    Isolated as a single seam so tests can replace pipeline construction by
    patching ``translation.core._load_pipeline`` — reliable, unlike patching
    ``transformers.pipeline`` (a lazy module attribute). Imported lazily so the
    heavy ``transformers`` import only happens when a model is actually loaded.
    """
    from transformers import pipeline

    return pipeline("translation", model=model_id, device=device)


def _login(token: str) -> None:
    """Authenticate with the HuggingFace Hub.

    A test seam (like :func:`_load_pipeline`): tests patch
    ``translation.core._login`` rather than ``huggingface_hub.login``, which is a
    lazy module attribute and cannot be patched reliably.
    """
    from huggingface_hub import login

    login(token)


class Translator:
    """Lazily loads HF translation pipelines and routes translation requests."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}
        self._device: int | None = None
        self._logged_in = False

    def _get_device(self) -> int:
        if self._device is None:
            import torch

            self._device = 0 if torch.cuda.is_available() else -1
        return self._device

    def _maybe_login(self) -> None:
        if self._logged_in:
            return
        token = get_hf_token()
        if token:
            _login(token)
        self._logged_in = True

    def _get_pipeline(self, model_id: str) -> Any:
        if model_id not in self._pipelines:
            self._maybe_login()
            self._pipelines[model_id] = _load_pipeline(model_id, self._get_device())
        return self._pipelines[model_id]

    def translate(
        self,
        text: Any,
        *,
        direction: str | None = None,
        model_id: str | None = None,
        **params: Any,
    ) -> Any:
        """Translate ``text`` using the model for ``direction`` or ``model_id``.

        Returns the raw pipeline prediction (list of ``{"translation_text": ...}``).
        Use :func:`extract_translations` to get plain strings.
        """
        resolved = resolve_model_id(direction=direction, model_id=model_id)
        pipe = self._get_pipeline(resolved)
        return pipe(text, **params)
