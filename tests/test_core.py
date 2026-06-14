import pytest

from translation.core import Translator, extract_translations


def test_translate_routes_to_correct_model():
    t = Translator()
    out = t.translate("Hello", direction="en-ng")
    # fake_load_pipeline embeds the resolved model id in its output
    assert out == [{"translation_text": "<meyabase/en-ng-translation>::Hello"}]


def test_translate_by_model_id():
    t = Translator()
    out = t.translate("Hi", model_id="meyabase/ng-en-translation")
    assert out == [{"translation_text": "<meyabase/ng-en-translation>::Hi"}]


def test_pipeline_is_cached(monkeypatch):
    import translation.core as core

    calls = []
    inner = core._load_pipeline  # the conftest fake, applied via autouse fixture

    def counting_load(model_id, device):
        calls.append(model_id)
        return inner(model_id, device)

    monkeypatch.setattr(core, "_load_pipeline", counting_load)
    t = Translator()
    t.translate("a", direction="en-ng")
    t.translate("b", direction="en-ng")
    assert calls == ["meyabase/en-ng-translation"]  # loaded once, reused


def test_translate_unknown_direction_raises():
    t = Translator()
    with pytest.raises(ValueError, match="Unknown direction"):
        t.translate("x", direction="zz-yy")


def test_login_called_when_token_set(monkeypatch):
    import translation.core as core

    monkeypatch.setenv("TF_TOKEN", "fake-token")
    calls = []
    monkeypatch.setattr(core, "_login", lambda token: calls.append(token))
    Translator().translate("Hello", direction="en-ng")
    assert calls == ["fake-token"]


def test_login_not_called_without_token(monkeypatch):
    import translation.core as core

    # TF_TOKEN is cleared by the autouse fixture; no login should be attempted.
    calls = []
    monkeypatch.setattr(core, "_login", lambda token: calls.append(token))
    Translator().translate("Hello", direction="en-ng")
    assert calls == []


def test_extract_translations_single():
    assert extract_translations([{"translation_text": "bonjour"}]) == "bonjour"


def test_extract_translations_multiple():
    pred = [{"translation_text": "a"}, {"translation_text": "b"}]
    assert extract_translations(pred) == ["a", "b"]


def test_extract_translations_dict():
    assert extract_translations({"translation_text": "solo"}) == "solo"
