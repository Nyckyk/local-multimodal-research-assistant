import types

import pytest

from scripts.regression_support import (
    ModelAvailabilityError,
    installed_ollama_models,
    require_configured_models,
)
from settings import OLLAMA_MODEL, VISION_MODEL


class FakeClient:
    def __init__(self, models=None, error=None):
        self.models = models or []
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return {"models": self.models}


def test_model_list_accepts_mapping_and_typed_rows():
    client = FakeClient([
        {"model": OLLAMA_MODEL},
        types.SimpleNamespace(model=VISION_MODEL),
    ])
    assert installed_ollama_models(client) == {OLLAMA_MODEL, VISION_MODEL}


def test_missing_model_error_is_concise_and_actionable():
    with pytest.raises(ModelAvailabilityError) as caught:
        require_configured_models(FakeClient([{"model": OLLAMA_MODEL}]))
    message = str(caught.value)
    assert VISION_MODEL in message
    assert f"ollama pull {VISION_MODEL}" in message
    assert "Traceback" not in message


def test_unavailable_ollama_error_is_concise():
    with pytest.raises(ModelAvailabilityError, match="Ollama is unavailable"):
        installed_ollama_models(FakeClient(error=ConnectionError("connection refused")))
