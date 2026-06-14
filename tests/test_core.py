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
