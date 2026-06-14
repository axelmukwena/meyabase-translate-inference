"""FastAPI server (non-HuggingFace entrypoint).

Run locally:  uv run uvicorn app:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from translation.config import DIRECTIONS
from translation.core import Translator, extract_translations

app = FastAPI(title="Meyabase Translation Inference")
_translator = Translator()


class TranslateRequest(BaseModel):
    text: str
    direction: str
    parameters: dict | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "directions": sorted(DIRECTIONS)}


@app.post("/translate")
def translate(req: TranslateRequest) -> dict:
    try:
        prediction = _translator.translate(
            req.text, direction=req.direction, **(req.parameters or {})
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"translation": extract_translations(prediction)}
