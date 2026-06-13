# Meyabase Translate Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine the two fine-tuned MarianMT models and the HF custom handler into one code-only GitHub repo with a HuggingFace entrypoint plus a local FastAPI server and CLI, all sharing one inference core.

**Architecture:** A `translation/` package holds the shared core (`config.py` for the direction→model-id registry, `core.py` for the `Translator` that lazily loads HF pipelines). Three thin entrypoints delegate to it: `handler.py` (HF `EndpointHandler`), `app.py` (FastAPI), `cli.py` (CLI). Models download from the Hub at runtime; no weights are committed.

**Tech Stack:** Python 3.10+, uv (deps + lockfile), transformers/torch, FastAPI/uvicorn/pydantic, pytest (mocked unit tests).

---

## File Structure

- `pyproject.toml` — uv project (deps, dev group, pytest config). No build-system; `package = false`.
- `.gitignore` — ignores model snapshot dirs, `.env`, caches, `.venv`.
- `.env.example` — `TF_TOKEN=` template.
- `translation/__init__.py` — package marker.
- `translation/config.py` — `DIRECTIONS` registry, `resolve_model_id`, `get_hf_token`.
- `translation/core.py` — `Translator` class, `extract_translations` helper.
- `handler.py` — HF `EndpointHandler` (root, HF requirement).
- `app.py` — FastAPI app (`POST /translate`, `GET /health`).
- `cli.py` — argparse CLI.
- `tests/conftest.py` — autouse pipeline/device mock (skipped for integration).
- `tests/test_config.py`, `test_core.py`, `test_handler.py`, `test_app.py`, `test_cli.py`, `test_integration.py`.
- `requirements.txt` — exported from `uv.lock` for HF Inference Endpoints.
- `README.md` — setup + run-locally + deploy notes.

---

## Task 1: Project scaffolding (uv + ignores)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `translation/__init__.py`

- [ ] **Step 1: Write `.gitignore` first (so the ~1.6GB model dirs are never staged)**

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/

# Env
.env

# Model snapshots (loaded from the Hub at runtime, not committed)
en-ng-translation/
ng-en-translation/
multi-translation-inference/

# OS
.DS_Store
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "meyabase-translate-inference"
version = "0.1.0"
description = "Inference for meyabase English<->Nigerian translation models, with HuggingFace and local entrypoints."
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "huggingface-hub>=0.23",
    "sentencepiece>=0.2",
    "sacremoses>=0.1",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pydantic>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]

[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "integration: real-model tests that download from the Hub (deselected by default)",
]
addopts = "-m 'not integration'"
```

- [ ] **Step 3: Create `.env.example`**

```dotenv
# HuggingFace token (optional). Needed for private/gated models or higher download rate limits.
TF_TOKEN=
```

- [ ] **Step 4: Create the package marker `translation/__init__.py`**

```python
"""Shared inference core for meyabase translation models."""
```

- [ ] **Step 5: Resolve the environment and verify it builds**

Run: `uv sync`
Expected: creates `.venv` and `uv.lock`, installs torch/transformers/fastapi/pytest with no errors. (First run downloads torch — may take a few minutes.)

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml uv.lock .env.example translation/__init__.py
git commit -m "chore: scaffold uv project and gitignore model snapshots"
```

---

## Task 2: Direction registry (`translation/config.py`)

**Files:**
- Create: `translation/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import pytest

from translation.config import DIRECTIONS, resolve_model_id


def test_directions_registry():
    assert DIRECTIONS["en-ng"] == "meyabase/en-ng-translation"
    assert DIRECTIONS["ng-en"] == "meyabase/ng-en-translation"


def test_resolve_by_direction():
    assert resolve_model_id(direction="en-ng") == "meyabase/en-ng-translation"


def test_resolve_by_model_id_passthrough():
    assert resolve_model_id(model_id="meyabase/ng-en-translation") == "meyabase/ng-en-translation"


def test_resolve_unknown_direction_raises():
    with pytest.raises(ValueError, match="Unknown direction"):
        resolve_model_id(direction="fr-en")


def test_resolve_unknown_model_id_raises():
    with pytest.raises(ValueError, match="Unknown model_id"):
        resolve_model_id(model_id="meyabase/bogus")


def test_resolve_requires_one_argument():
    with pytest.raises(ValueError, match="direction or model_id"):
        resolve_model_id()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'translation.config'`.

- [ ] **Step 3: Implement `translation/config.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add translation/config.py tests/test_config.py
git commit -m "feat: add direction registry and model-id resolution"
```

---

## Task 3: Inference core (`translation/core.py`)

**Files:**
- Create: `translation/core.py`
- Create: `tests/conftest.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Create the autouse mock fixture `tests/conftest.py`**

This patches `transformers.pipeline` and torch device detection so unit tests never download a model. It is skipped for tests marked `integration`.

```python
import pytest


@pytest.fixture(autouse=True)
def mock_pipeline(request, monkeypatch):
    """Replace transformers.pipeline and torch device detection (no model downloads).

    Skipped for tests marked `integration`, which use the real pipeline.
    """
    if request.node.get_closest_marker("integration"):
        return

    import torch
    import transformers

    monkeypatch.delenv("TF_TOKEN", raising=False)

    def fake_pipeline(task, model=None, device=None, **kwargs):
        def run(text, **params):
            if isinstance(text, list):
                return [{"translation_text": f"<{model}>::{t}"} for t in text]
            return [{"translation_text": f"<{model}>::{text}"}]
        return run

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_core.py`:

```python
import pytest

from translation.core import Translator, extract_translations


def test_translate_routes_to_correct_model():
    t = Translator()
    out = t.translate("Hello", direction="en-ng")
    # fake_pipeline embeds the resolved model id in its output
    assert out == [{"translation_text": "<meyabase/en-ng-translation>::Hello"}]


def test_translate_by_model_id():
    t = Translator()
    out = t.translate("Hi", model_id="meyabase/ng-en-translation")
    assert out == [{"translation_text": "<meyabase/ng-en-translation>::Hi"}]


def test_pipeline_is_cached(monkeypatch):
    import transformers

    calls = []
    orig = transformers.pipeline

    def counting_pipeline(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(transformers, "pipeline", counting_pipeline)
    t = Translator()
    t.translate("a", direction="en-ng")
    t.translate("b", direction="en-ng")
    assert calls == ["meyabase/en-ng-translation"]  # loaded once, reused


def test_translate_unknown_direction_raises():
    t = Translator()
    with pytest.raises(ValueError, match="Unknown direction"):
        t.translate("x", direction="zz-yy")


def test_extract_translations_single():
    assert extract_translations([{"translation_text": "bonjour"}]) == "bonjour"


def test_extract_translations_multiple():
    pred = [{"translation_text": "a"}, {"translation_text": "b"}]
    assert extract_translations(pred) == ["a", "b"]


def test_extract_translations_dict():
    assert extract_translations({"translation_text": "solo"}) == "solo"
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'translation.core'`.

- [ ] **Step 4: Implement `translation/core.py`**

```python
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
            from huggingface_hub import login

            login(token)
        self._logged_in = True

    def _get_pipeline(self, model_id: str) -> Any:
        if model_id not in self._pipelines:
            from transformers import pipeline

            self._maybe_login()
            self._pipelines[model_id] = pipeline(
                "translation", model=model_id, device=self._get_device()
            )
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
        return pipe(text, **params) if params else pipe(text)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_core.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add translation/core.py tests/conftest.py tests/test_core.py
git commit -m "feat: add Translator core with lazy pipeline loading"
```

---

## Task 4: HuggingFace entrypoint (`handler.py`)

**Files:**
- Create: `handler.py`
- Test: `tests/test_handler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_handler.py`:

```python
import pytest

from handler import EndpointHandler


def test_handler_translates_by_direction():
    h = EndpointHandler()
    out = h({"inputs": "Hello", "direction": "en-ng"})
    assert out == [{"translation_text": "<meyabase/en-ng-translation>::Hello"}]


def test_handler_translates_by_model_id_backward_compat():
    h = EndpointHandler()
    out = h({"inputs": "Hi", "model_id": "meyabase/ng-en-translation"})
    assert out == [{"translation_text": "<meyabase/ng-en-translation>::Hi"}]


def test_handler_missing_inputs_raises():
    h = EndpointHandler()
    with pytest.raises(ValueError, match="inputs"):
        h({"direction": "en-ng"})


def test_handler_invalid_direction_raises():
    h = EndpointHandler()
    with pytest.raises(ValueError, match="Unknown direction"):
        h({"inputs": "x", "direction": "zz-yy"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_handler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'handler'`.

- [ ] **Step 3: Implement `handler.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_handler.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add handler.py tests/test_handler.py
git commit -m "feat: add HuggingFace EndpointHandler entrypoint"
```

---

## Task 5: FastAPI entrypoint (`app.py`)

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "en-ng" in body["directions"]


def test_translate_ok():
    resp = client.post("/translate", json={"text": "Hello", "direction": "en-ng"})
    assert resp.status_code == 200
    assert resp.json() == {"translation": "<meyabase/en-ng-translation>::Hello"}


def test_translate_bad_direction_returns_400():
    resp = client.post("/translate", json={"text": "Hello", "direction": "zz-yy"})
    assert resp.status_code == 400
    assert "Unknown direction" in resp.json()["detail"]


def test_translate_missing_text_returns_422():
    resp = client.post("/translate", json={"direction": "en-ng"})
    assert resp.status_code == 422  # pydantic validation
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Implement `app.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add FastAPI server entrypoint"
```

---

## Task 6: CLI entrypoint (`cli.py`)

**Files:**
- Create: `cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import pytest

from cli import main


def test_cli_prints_translation(capsys):
    rc = main(["--direction", "en-ng", "Hello"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "<meyabase/en-ng-translation>::Hello"


def test_cli_invalid_direction_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--direction", "zz-yy", "Hello"])
    assert exc.value.code == 2  # argparse choices error


def test_cli_requires_direction(capsys):
    with pytest.raises(SystemExit):
        main(["Hello"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cli'`.

- [ ] **Step 3: Implement `cli.py`**

```python
"""Command-line entrypoint (non-HuggingFace).

Usage:  uv run cli.py --direction en-ng "Hello, how are you?"
"""
from __future__ import annotations

import argparse

from translation.config import DIRECTIONS
from translation.core import Translator, extract_translations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate text with meyabase models.")
    parser.add_argument("text", help="Text to translate.")
    parser.add_argument(
        "--direction",
        "-d",
        required=True,
        choices=sorted(DIRECTIONS),
        help="Translation direction.",
    )
    args = parser.parse_args(argv)

    translator = Translator()
    prediction = translator.translate(args.text, direction=args.direction)
    print(extract_translations(prediction))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: add CLI entrypoint"
```

---

## Task 7: Real-model integration test (skipped by default)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Create `tests/test_integration.py`**

```python
import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to download models and run",
)
def test_real_en_ng_translation():
    from translation.core import Translator, extract_translations

    translator = Translator()
    out = extract_translations(translator.translate("Hello, how are you?", direction="en-ng"))
    assert isinstance(out, str) and out.strip()
```

- [ ] **Step 2: Verify it is deselected by default**

Run: `uv run pytest tests/test_integration.py -v`
Expected: `1 deselected` (the `addopts = -m 'not integration'` filter excludes it; no download happens).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add skipped real-model integration test"
```

---

## Task 8: README and exported requirements.txt

**Files:**
- Create: `README.md`
- Create: `requirements.txt` (generated)

- [ ] **Step 1: Generate `requirements.txt` for HF Inference Endpoints**

Run: `uv export --no-hashes --no-dev -o requirements.txt`
Expected: writes `requirements.txt` listing torch/transformers/fastapi/etc. (no dev deps, no hashes).

- [ ] **Step 2: Create `README.md`**

````markdown
# meyabase-translate-inference

Inference for the meyabase English↔Nigerian MarianMT translation models, with two
entrypoints:

- **HuggingFace** — a custom `EndpointHandler` (`handler.py`) for HF Inference Endpoints.
- **Local** — a FastAPI server (`app.py`) and a CLI (`cli.py`).

Models are downloaded from the Hub at runtime (`meyabase/en-ng-translation`,
`meyabase/ng-en-translation`); no weights are stored in this repo.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Optional: a HuggingFace token, for private/gated models or higher rate limits.

## Setup

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # then add your TF_TOKEN if you have one (optional)
```

`TF_TOKEN` is read from the environment. Export it or put it in `.env` and load it
(`export $(grep -v '^#' .env | xargs)`).

## Run locally

### FastAPI server

```bash
uv run uvicorn app:app --reload
```

Then in another terminal:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/translate \
  -H 'content-type: application/json' \
  -d '{"text": "Hello, how are you?", "direction": "en-ng"}'
# -> {"translation": "..."}
```

Directions: `en-ng` (English→Nigerian) and `ng-en` (Nigerian→English).

### CLI

```bash
uv run cli.py --direction en-ng "Hello, how are you?"
uv run cli.py -d ng-en "..."
```

> The first request for a direction downloads that model from the Hub, so it is
> slower; later requests reuse the cached, in-memory pipeline.

## Tests

```bash
uv run pytest                                 # fast, mocked — no downloads
RUN_INTEGRATION=1 uv run pytest -m integration  # real models (downloads from the Hub)
```

## Deploy as a HuggingFace Inference Endpoint

HF Inference Endpoints install dependencies from `requirements.txt` (not uv). Point
the endpoint at a repo containing `handler.py` + `requirements.txt`. Regenerate the
file whenever dependencies change:

```bash
uv export --no-hashes --no-dev -o requirements.txt
```

Request payload:

```json
{"inputs": "Hello", "direction": "en-ng", "parameters": {}}
```

`model_id` (e.g. `"meyabase/en-ng-translation"`) is accepted instead of `direction`
for backward compatibility.
````

- [ ] **Step 3: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: add README and exported HF requirements.txt"
```

---

## Task 9: Full verification and GitHub push

**Files:** none (verification + remote setup)

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest -v`
Expected: all unit tests pass (config 6, core 7, handler 4, app 4, cli 3 = 24 passed), integration deselected.

- [ ] **Step 2: Smoke-test the entrypoints import cleanly**

Run: `uv run python -c "import handler, app, cli; print('entrypoints import OK')"`
Expected: prints `entrypoints import OK` (no model download — nothing is invoked).

- [ ] **Step 3: Confirm no large files are tracked by git**

Run: `git ls-files | grep -E '\.(bin|pt|pth|spm)$' || echo "no model artifacts tracked"`
Expected: `no model artifacts tracked`.

- [ ] **Step 4: Create the GitHub repo and push (asks the user first)**

Confirm the repo name/visibility with the user, then:

```bash
gh repo create meyabase-translate-inference --private --source=. --remote=origin --push
```

Expected: repo created and `main` pushed. (Use `--public` instead of `--private` if the user wants it public.)

- [ ] **Step 5: Verify the remote**

Run: `git remote -v && gh repo view --web`
Expected: `origin` points at the new GitHub repo.

---

## Notes for the implementer

- Run every command with `uv run ...` so it uses the project venv.
- The autouse `mock_pipeline` fixture means unit tests never hit the network; only the integration test (behind `RUN_INTEGRATION=1`) downloads models.
- Do not `git add` the `en-ng-translation/`, `ng-en-translation/`, or `multi-translation-inference/` directories — `.gitignore` covers them, but double-check with `git status` before each commit.
