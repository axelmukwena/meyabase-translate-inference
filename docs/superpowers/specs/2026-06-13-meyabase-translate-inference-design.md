# Design: meyabase-translate-inference

**Date:** 2026-06-13
**Status:** Approved (design); pending spec review

## Goal

Combine the existing pieces — two fine-tuned MarianMT translation models and a
HuggingFace custom inference handler — into a single, testable, GitHub-ready
repository with two ways to run inference:

1. **HuggingFace entrypoint** — the custom `EndpointHandler` used by HF Inference
   Endpoints.
2. **Non-HuggingFace entrypoint** — a local FastAPI server and a CLI.

Plus a README documenting how to spin it up and run locally.

## Decisions (from brainstorming)

- **Model weights:** Not committed. Both entrypoints download the models from the
  Hub at runtime (`meyabase/en-ng-translation`, `meyabase/ng-en-translation`).
  Repo is code-only. This matches what the original `handler.py` already does.
- **Local entrypoint:** FastAPI server (`POST /translate`) **and** a CLI, both
  sharing one core inference module.
- **Testing:** Fast unit tests with the HF `pipeline` mocked (no downloads). One
  real-model integration test included but skipped by default behind an env flag.
- **Dependency management:** `uv` (`pyproject.toml` + `uv.lock`) is the primary
  workflow. A `requirements.txt` is exported from the lockfile and kept in the
  repo **solely** for HF Inference Endpoints, which only read `requirements.txt`.

## Repository layout

```
meyabase-translate-inference/
├── README.md                 # setup + run-locally notes (both entrypoints)
├── pyproject.toml            # uv project: deps, dev group, pytest/ruff config
├── uv.lock                   # locked dependency versions
├── requirements.txt          # EXPORTED from uv.lock; for HF Inference Endpoints only
├── .gitignore                # model snapshot dirs, .env, caches, .venv
├── .env.example              # TF_TOKEN=...
├── translation/              # shared core (single source of truth)
│   ├── __init__.py
│   ├── config.py             # DIRECTIONS registry → model_id; env handling
│   └── core.py               # Translator: lazy-load pipelines, translate(...)
├── handler.py                # HF entrypoint: EndpointHandler → delegates to Translator
├── app.py                    # non-HF entrypoint: FastAPI, POST /translate, GET /health
├── cli.py                    # non-HF entrypoint: python cli.py --direction en-ng "..."
├── docs/superpowers/specs/   # this design doc
└── tests/
    ├── conftest.py
    ├── test_core.py
    ├── test_handler.py
    ├── test_app.py
    └── test_cli.py
```

The existing on-disk folders `en-ng-translation/`, `ng-en-translation/`, and
`multi-translation-inference/` are **gitignored** (kept locally, not committed).
The logic from `multi-translation-inference/handler.py` is migrated into the new
structure.

## Components and data flow

### `translation/config.py`
- `DIRECTIONS`: mapping of friendly direction key → Hub model id.
  - `"en-ng"` → `meyabase/en-ng-translation`
  - `"ng-en"` → `meyabase/ng-en-translation`
- Helper to resolve a request to a model id, accepting either a `direction`
  (`"en-ng"`) or a full `model_id` (`"meyabase/en-ng-translation"`) for backward
  compatibility with existing HF callers.
- Reads `TF_TOKEN` from the environment (optional). One place to add future pairs.

### `translation/core.py` — `Translator`
- The only module that imports `transformers` / `torch`.
- Detects device (GPU if available, else CPU).
- Lazily loads a `pipeline("translation", model_id, device=...)` per direction on
  first use; caches loaded pipelines. Lazy loading keeps startup cheap and lets
  tests run without downloads.
- `translate(text, direction, **params) -> str` (or list for batch input).
- Raises a clear `ValueError` (listing valid directions) on unknown
  direction/model id.

### `handler.py` — HF entrypoint
- `EndpointHandler` with the HF contract: `__init__(self, path="")` and
  `__call__(self, data)`.
- Accepts the original payload shape (`inputs`, `parameters`, `model_id`) **and**
  a friendlier `direction` field. Delegates to `Translator`.
- Lives at repo root (HF requirement), importing the `translation` package.

### `app.py` — FastAPI entrypoint
- `POST /translate` with body `{text, direction, parameters?}` → `{translation}`.
- `GET /health` → `{status: "ok"}`.
- Constructs one shared `Translator`. Run: `uv run uvicorn app:app --reload`.

### `cli.py` — CLI entrypoint
- `python cli.py --direction en-ng "Hello"` (via `uv run cli.py ...`) → prints
  translation. Uses the same `Translator`.

## Error handling

- Unknown direction / model id → `ValueError` listing valid directions (HF +
  core); surfaced as HTTP 400 in the API.
- Missing `text`/`inputs` → HTTP 400 in the API; clear error in CLI/handler.
- Missing `TF_TOKEN` → proceed with a warning (model load is attempted
  regardless; public models may still load).

## Testing

- `uv run pytest`. HF `pipeline` is mocked in `conftest.py` so no model is
  downloaded.
- Coverage: direction/model-id resolution, valid + invalid inputs, `Translator`
  routing and caching, `EndpointHandler` payload handling, FastAPI endpoints via
  `TestClient`, CLI argument parsing and output.
- One real-model integration test marked and skipped unless an env flag
  (e.g. `RUN_INTEGRATION=1`) is set.

## README contents

- Prereqs: Python, `uv`, optional `TF_TOKEN`.
- Install: `uv sync`.
- Run local API: `uv run uvicorn app:app --reload` + a `curl` example.
- Run CLI: `uv run cli.py --direction en-ng "Hello"`.
- Run tests: `uv run pytest`.
- Regenerate HF `requirements.txt`: `uv export --no-hashes --no-dev -o requirements.txt`.
- Deploy as HF Inference Endpoint: point the endpoint at a repo containing
  `handler.py` + `requirements.txt`; note the expected request payload.

## Out of scope

- Committing or hosting model weights (loaded from the Hub).
- Training / fine-tuning code (this repo is inference-only).
- Authentication / rate limiting on the local API.
- Adding new language pairs (the config makes this easy later, but none added now).
