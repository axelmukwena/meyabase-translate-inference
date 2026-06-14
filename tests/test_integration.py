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
