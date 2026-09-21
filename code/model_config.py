"""Private, provider-neutral transport settings for teacher-model requests.

The credential file is deliberately outside the repository and output tree.
Only its endpoint and key are read here; callers still choose the model name.
Legacy OPENROUTER_* variables remain a fallback for old, explicitly launched
jobs, but the private teacher configuration has higher priority.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CONFIG_PATH = Path.home() / ".config/agentwm/teacher_provider.json"
DEFAULT_OPENROUTER_CONFIG_PATH = Path.home() / ".config/agentwm/openrouter_provider.json"


def config_path() -> Path:
    value = os.getenv("AGENTWM_TEACHER_PROVIDER_CONFIG", "").strip()
    return Path(value).expanduser() if value else DEFAULT_CONFIG_PATH


def _private_config() -> dict[str, Any]:
    path = config_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _openrouter_private_config() -> dict[str, Any]:
    """Read the separate private credential reserved for Qwen review calls."""
    configured_path = os.getenv("AGENTWM_OPENROUTER_PROVIDER_CONFIG", "").strip()
    path = Path(configured_path).expanduser() if configured_path else DEFAULT_OPENROUTER_CONFIG_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def teacher_api_key() -> str:
    """Return the configured teacher credential without logging it."""
    direct = os.getenv("AGENTWM_TEACHER_API_KEY", "").strip()
    if direct:
        return direct
    configured = _private_config().get("api_key")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def teacher_base_url(default: str = DEFAULT_BASE_URL) -> str:
    """Return the OpenAI-compatible teacher API base URL without a trailing slash."""
    direct = os.getenv("AGENTWM_TEACHER_BASE_URL", "").strip()
    if direct:
        return direct.rstrip("/")
    configured = _private_config().get("base_url")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().rstrip("/")
    legacy = os.getenv("OPENROUTER_BASE_URL", "").strip()
    return (legacy or default).rstrip("/")


def openrouter_api_key() -> str:
    """Return the credential reserved for Qwen filter/safety calls.

    Teacher traffic may use a separate provider (for example Ark), but the
    lightweight Qwen reviewer remains on OpenRouter.  Keeping this lookup
    independent prevents a proxy child from accidentally sending an Ark key
    to OpenRouter or vice versa.
    """
    direct = os.getenv("AGENTWM_OPENROUTER_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    if direct:
        return direct
    configured = _openrouter_private_config().get("api_key")
    return configured.strip() if isinstance(configured, str) else ""


def openrouter_base_url(default: str = DEFAULT_BASE_URL) -> str:
    """Return the OpenRouter-compatible endpoint for Qwen filter/safety."""
    direct = os.getenv("AGENTWM_OPENROUTER_BASE_URL", "").strip() or os.getenv("OPENROUTER_BASE_URL", "").strip()
    if direct:
        return direct.rstrip("/")
    configured = _openrouter_private_config().get("base_url")
    return (configured.strip() if isinstance(configured, str) and configured.strip() else default).rstrip("/")


def auxiliary_api_key() -> str:
    """Return the credential for filter/safety auxiliary-model calls.

    Existing jobs keep using the separate OpenRouter credential by default.
    A launcher can route the auxiliary model through another OpenAI-compatible
    provider (for example ModelArk) with ``AGENTWM_AUX_API_KEY``.
    """
    direct = os.getenv("AGENTWM_AUX_API_KEY", "").strip()
    return direct or openrouter_api_key()


def auxiliary_base_url(default: str = DEFAULT_BASE_URL) -> str:
    """Return the OpenAI-compatible endpoint for auxiliary-model calls."""
    direct = os.getenv("AGENTWM_AUX_BASE_URL", "").strip()
    return direct.rstrip("/") if direct else openrouter_base_url(default)
