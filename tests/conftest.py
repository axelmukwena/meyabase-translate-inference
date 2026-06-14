import pytest


@pytest.fixture(autouse=True)
def mock_pipeline(request, monkeypatch):
    """Replace pipeline construction so unit tests never download a model.

    Patches ``translation.core._load_pipeline`` (our own seam) rather than
    ``transformers.pipeline``: transformers is a lazy module whose attribute
    cannot be patched reliably. Skipped for tests marked `integration`, which
    use the real pipeline.
    """
    if request.node.get_closest_marker("integration"):
        return

    import translation.core as core

    monkeypatch.delenv("TF_TOKEN", raising=False)

    def fake_load_pipeline(model_id, device):
        def run(text, **params):
            if isinstance(text, list):
                return [{"translation_text": f"<{model_id}>::{t}"} for t in text]
            return [{"translation_text": f"<{model_id}>::{text}"}]
        return run

    monkeypatch.setattr(core, "_load_pipeline", fake_load_pipeline)
