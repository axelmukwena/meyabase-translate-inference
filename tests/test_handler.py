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
