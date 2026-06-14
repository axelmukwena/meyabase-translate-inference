import pytest

from translation.config import DIRECTIONS, get_hf_token, resolve_model_id


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


def test_get_hf_token_returns_token(monkeypatch):
    monkeypatch.setenv("TF_TOKEN", "hf_abc123")
    assert get_hf_token() == "hf_abc123"


def test_get_hf_token_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("TF_TOKEN", raising=False)
    assert get_hf_token() is None
