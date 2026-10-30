"""Thin client between the Streamlit UI and the FastAPI backend (Person 5).

The UI never talks to the model directly. Everything the UI needs goes through
this module so the API contract lives in ONE place. If the contract changes,
only this file changes.

Contract (see Project Documentation, section 8.3):
    GET  {API_URL}/health   -> 200 when the service is up
    POST {API_URL}/predict  -> multipart form, field name "file"  (agree with Person 5)
"""
from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass, replace
from typing import Any

import requests

DEFAULT_DISCLAIMER = (
    "AI-generated prediction; it may be incorrect and is not a medical diagnosis."
)
SUPPORTED_EXTENSIONS = (".wav",)  # start with WAV only; add formats once tested


class ApiError(Exception):
    """An error that is safe to show to end users (no stack traces, no secrets)."""


@dataclass(frozen=True)
class Settings:
    api_url: str = "http://localhost:8000"
    predict_path: str = "/predict"
    health_path: str = "/health"
    timeout_seconds: float = 60.0
    max_upload_mb: float = 10.0
    max_duration_seconds: float = 30.0
    mock: bool = False


def load_settings() -> Settings:
    """Read configuration from environment variables (no hardcoded URLs/paths)."""
    d = Settings()
    return Settings(
        api_url=os.getenv("DYSARTHRIA_API_URL", d.api_url).rstrip("/"),
        predict_path=os.getenv("DYSARTHRIA_PREDICT_PATH", d.predict_path),
        health_path=os.getenv("DYSARTHRIA_HEALTH_PATH", d.health_path),
        timeout_seconds=float(os.getenv("DYSARTHRIA_TIMEOUT_SECONDS", d.timeout_seconds)),
        max_upload_mb=float(os.getenv("DYSARTHRIA_MAX_UPLOAD_MB", d.max_upload_mb)),
        max_duration_seconds=float(
            os.getenv("DYSARTHRIA_MAX_DURATION_SECONDS", d.max_duration_seconds)
        ),
        mock=os.getenv("DYSARTHRIA_MOCK", "0").lower() in ("1", "true", "yes"),
    )


def with_mock(settings: Settings, mock: bool) -> Settings:
    return replace(settings, mock=mock)


# --------------------------------------------------------------------------- #
# Upload validation (UI-side, so users get fast feedback; backend validates too)
# --------------------------------------------------------------------------- #
def wav_duration_seconds(data: bytes) -> float | None:
    """Return the duration of a WAV file, or None if it can't be decoded."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, EOFError):
        return None


def validate_upload(filename: str, data: bytes, settings: Settings) -> str | None:
    """Return a user-facing error message, or None if the file looks fine."""
    if not data:
        return "This file is empty. Choose a recording that contains audio."
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        return "This file type isn't supported yet. Upload a .wav recording."
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        return (
            f"This file is {size_mb:.1f} MB. The limit is "
            f"{settings.max_upload_mb:g} MB. Trim the recording and try again."
        )
    duration = wav_duration_seconds(data)
    if duration is None:
        return "This WAV file couldn't be read. Re-export it and try again."
    if duration <= 0:
        return "This recording has no audio in it."
    if duration > settings.max_duration_seconds:
        return (
            f"This recording is {duration:.0f} seconds. The limit is "
            f"{settings.max_duration_seconds:g} seconds."
        )
    return None


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #
def check_health(settings: Settings) -> tuple[bool, str]:
    try:
        r = requests.get(settings.api_url + settings.health_path, timeout=5)
        if r.ok:
            return True, "Connected to the analysis service."
        return False, f"The service answered with status {r.status_code}."
    except requests.RequestException:
        return False, f"Can't reach the service at {settings.api_url}."


def normalize_response(payload: Any) -> dict:
    """Make the response safe for the UI: fixed keys, honest defaults.

    We never invent values. Missing confidence stays None; missing explanation
    becomes available=False.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("transcript"), str):
        raise ApiError("The service sent a response this app can't read.")
    expl = payload.get("explanation")
    if not isinstance(expl, dict):
        expl = {}
    return {
        "transcript": payload["transcript"],
        "confidence": payload.get("confidence"),
        "confidence_type": payload.get("confidence_type"),
        "model_version": payload.get("model_version") or "unknown",
        "warnings": [str(w) for w in (payload.get("warnings") or [])],
        "explanation": {
            "available": bool(expl.get("available", False)),
            "method": expl.get("method"),
            "visualization_url": expl.get("visualization_url"),
            "summary": expl.get("summary"),
            "limitations": [str(x) for x in (expl.get("limitations") or [])],
        },
        "disclaimer": payload.get("disclaimer") or DEFAULT_DISCLAIMER,
        "is_mock": False,
    }


def mock_response() -> dict:
    """Development-only response. Clearly labelled; never a real prediction."""
    return {
        "transcript": "[mock transcript: no model was used]",
        "confidence": None,
        "confidence_type": None,
        "model_version": "mock",
        "warnings": [],
        "explanation": {
            "available": False,
            "method": None,
            "visualization_url": None,
            "summary": None,
            "limitations": ["Demo mode: no explanation was generated."],
        },
        "disclaimer": DEFAULT_DISCLAIMER,
        "is_mock": True,
    }


def predict(filename: str, data: bytes, settings: Settings) -> dict:
    """Send audio to the backend and return a normalized response dict."""
    if settings.mock:
        return mock_response()

    url = settings.api_url + settings.predict_path
    try:
        r = requests.post(
            url,
            files={"file": (filename, data, "audio/wav")},
            timeout=settings.timeout_seconds,
        )
    except requests.Timeout:
        raise ApiError("The analysis took too long. Try a shorter recording.") from None
    except requests.ConnectionError:
        raise ApiError(
            "Can't reach the analysis service. Check that the backend is running."
        ) from None
    except requests.RequestException:
        raise ApiError("The request failed. Try again in a moment.") from None

    if not r.ok:
        # Show the backend's controlled message if it gave one, never raw bodies.
        message = None
        try:
            body = r.json()
            if isinstance(body, dict):
                message = body.get("message") or body.get("detail")
        except ValueError:
            pass
        if isinstance(message, str) and message:
            raise ApiError(message)
        raise ApiError(f"The service couldn't process this file (status {r.status_code}).")

    try:
        payload = r.json()
    except ValueError:
        raise ApiError("The service sent a response this app can't read.") from None
    if isinstance(payload, dict) and payload.get("status") not in (None, "success"):
        raise ApiError(str(payload.get("message") or "The analysis didn't complete."))
    return normalize_response(payload)
