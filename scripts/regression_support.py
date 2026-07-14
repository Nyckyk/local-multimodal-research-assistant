"""Shared preflight helpers for the local regression suite."""

from __future__ import annotations

from dataclasses import dataclass

import ollama

from settings import OLLAMA_MODEL, VISION_MODEL


@dataclass
class ModelAvailabilityError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def _model_name(item) -> str:
    if isinstance(item, dict):
        return str(item.get("model") or item.get("name") or "")
    return str(getattr(item, "model", None) or getattr(item, "name", None) or "")


def installed_ollama_models(client=ollama) -> set[str]:
    """Return exact model names from the local Ollama list API."""
    try:
        response = client.list()
    except Exception as error:
        raise ModelAvailabilityError(
            "Ollama is unavailable. Start Ollama, then rerun the regression suite. "
            f"Detail: {error}"
        ) from None
    models = response.get("models", []) if isinstance(response, dict) else response.models
    return {name for item in models if (name := _model_name(item))}


def require_configured_models(client=ollama) -> set[str]:
    installed = installed_ollama_models(client)
    for model in (OLLAMA_MODEL, VISION_MODEL):
        if model not in installed:
            raise ModelAvailabilityError(
                f"Missing configured Ollama model: {model}. Pull it with: ollama pull {model}"
            )
    return installed
