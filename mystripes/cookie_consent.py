from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any, Mapping

COOKIE_CONSENT_COOKIE_NAME = "mystripes_cookie_consent_v1"
COOKIE_CONSENT_MAX_AGE_SECONDS = 60 * 60 * 24 * 180
COOKIE_CONSENT_VERSION = 1
OPTIONAL_CONVENIENCE_PURPOSE = "storyline_convenience"


def normalize_cookie_consent_choice(choice: object) -> str:
    normalized = str(choice or "").strip().lower()
    if normalized not in {"accepted", "rejected"}:
        raise ValueError("Cookie consent choice must be 'accepted' or 'rejected'.")
    return normalized


def build_cookie_consent_payload(
    choice: object,
    *,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    normalized_choice = normalize_cookie_consent_choice(choice)
    timestamp = updated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    timestamp = timestamp.replace(microsecond=0)
    return {
        "version": COOKIE_CONSENT_VERSION,
        "choice": normalized_choice,
        "updated_at": timestamp.isoformat().replace("+00:00", "Z"),
        "purposes": {
            OPTIONAL_CONVENIENCE_PURPOSE: normalized_choice == "accepted",
        },
    }


def normalize_cookie_consent_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Cookie consent data must be a mapping.")

    normalized_choice = normalize_cookie_consent_choice(payload.get("choice"))
    updated_at = str(payload.get("updated_at", "")).strip()
    if not updated_at:
        raise ValueError("Cookie consent data is missing updated_at.")

    try:
        datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Cookie consent updated_at must be an ISO-8601 timestamp.") from exc

    return {
        "version": int(payload.get("version", COOKIE_CONSENT_VERSION)),
        "choice": normalized_choice,
        "updated_at": updated_at,
        "purposes": {
            OPTIONAL_CONVENIENCE_PURPOSE: normalized_choice == "accepted",
        },
    }


def encode_cookie_consent_value(payload_or_choice: Mapping[str, Any] | object) -> str:
    if isinstance(payload_or_choice, Mapping):
        normalized = normalize_cookie_consent_payload(payload_or_choice)
    else:
        normalized = build_cookie_consent_payload(payload_or_choice)

    raw_json = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw_json).decode("ascii").rstrip("=")


def decode_cookie_consent_value(value: str) -> dict[str, Any]:
    if not value:
        raise ValueError("Missing cookie consent value.")

    try:
        padding = "=" * (-len(value) % 4)
        raw_json = base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii")).decode("utf-8")
        return normalize_cookie_consent_payload(json.loads(raw_json))
    except Exception as exc:
        raise ValueError("Invalid cookie consent value.") from exc


def cookie_consent_choice(payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        return normalize_cookie_consent_choice(payload.get("choice"))
    except ValueError:
        return None


def optional_cookie_consent_granted(payload: Mapping[str, Any] | None) -> bool:
    return cookie_consent_choice(payload) == "accepted"
