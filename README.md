# meyabase-translate-inference

Inference for the meyabase translation models, with two entrypoints:

- **HuggingFace** — a custom `EndpointHandler` ([handler.py](handler.py)) for HF Inference Endpoints.
- **Local** — a FastAPI server ([app.py](app.py)) and a CLI ([cli.py](cli.py)).

Two directions are served, both fine-tuned from the Helsinki-NLP `opus-mt-en-ng`
base model:

| Direction | Model | Source → Target |
|-----------|-------|-----------------|
| `en-ng`   | `meyabase/en-ng-translation` | English → `ng` |
| `ng-en`   | `meyabase/ng-en-translation` | `ng` → English |

Models are downloaded from the Hub at runtime; no weights are stored in this repo.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Optional: a HuggingFace token, for private/gated models or higher rate limits.

## Setup

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # then add your TF_TOKEN if you have one (optional)
```

`TF_TOKEN` is read from the environment. Export it, or load it from `.env`:

```bash
export $(grep -v '^#' .env | xargs)
```

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

`POST /translate` accepts `{"text": ..., "direction": "en-ng" | "ng-en", "parameters": {...}?}`.
Unknown directions return HTTP 400; a missing `text` returns HTTP 422.

### CLI

```bash
uv run cli.py --direction en-ng "Hello, how are you?"
uv run cli.py -d ng-en "Ndapandula unene"
```

> The first request for a direction downloads that model from the Hub, so it is
> slower; later requests reuse the cached, in-memory pipeline.

## Tests

```bash
uv run pytest                                       # fast, mocked — no downloads
RUN_INTEGRATION=1 uv run pytest -m integration      # real models (downloads from the Hub)
```

The unit tests mock pipeline construction (via the `translation.core._load_pipeline`
seam), so they run instantly and never hit the network. The integration test is
deselected by default and only runs with `RUN_INTEGRATION=1`.

## Deploy as a HuggingFace Inference Endpoint

HF Inference Endpoints install dependencies from `requirements.txt` (not uv). Point
the endpoint at a repo containing [handler.py](handler.py) + `requirements.txt`.
Regenerate the file whenever dependencies change:

```bash
uv export --no-hashes --no-dev -o requirements.txt
```

Request payload:

```json
{"inputs": "Hello", "direction": "en-ng", "parameters": {}}
```

`model_id` (e.g. `"meyabase/en-ng-translation"`) is accepted instead of `direction`
for backward compatibility with existing callers.

## Project layout

```
translation/        shared inference core
  config.py         direction → model-id registry, token handling
  core.py           Translator: lazy-load + cache pipelines, translate()
handler.py          HuggingFace EndpointHandler (HF entrypoint)
app.py              FastAPI server (local entrypoint)
cli.py              CLI (local entrypoint)
tests/              fast mocked unit tests + a skipped integration test
```
