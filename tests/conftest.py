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
