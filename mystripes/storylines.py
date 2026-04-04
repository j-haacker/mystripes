from __future__ import annotations

import base64
import hashlib
import json
import zlib
from datetime import date
from pathlib import Path
from typing import Any, Mapping

LOCAL_STORYLINES_PATH = Path(".streamlit/local_storylines.json")
STORYLINE_COOKIE_PREFIX = "mystripes_storyline_v1_"
STORYLINE_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365 * 3
STORYLINE_COOKIE_MAX_VALUE_LENGTH = 3500


def storyline_storage_backend_from_host(host: str | None) -> str:
    normalized_host = str(host or "").strip().lower().split(":", 1)[0]
    if normalized_host.endswith(".streamlit.app") or normalized_host.endswith(".streamlitapp.com"):
        return "cookie"
    return "file"


def normalize_storyline_name(name: object) -> str:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        raise ValueError("Enter a story line name before saving.")
    return normalized_name


def serialize_storyline_payload(
    name: object,
    birth_date: date,
    period_entries: list[Mapping[str, Any]],
    *,
    include_boundary_geojson: bool,
) -> dict[str, Any]:
    storyline_name = normalize_storyline_name(name)
    return {
        "version": 1,
        "name": storyline_name,
        "birth_date": birth_date.isoformat(),
        "period_entries": [
            _serialize_period_entry(entry, include_boundary_geojson=include_boundary_geojson)
            for entry in period_entries
        ],
    }


def normalize_storyline_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Story line data must be a mapping.")

    raw_birth_date = payload.get("birth_date")
    if not raw_birth_date:
        raise ValueError("Story line data is missing a birth_date.")

    raw_entries = payload.get("period_entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Story line data must include at least one period entry.")

    return {
        "version": int(payload.get("version", 1)),
        "name": normalize_storyline_name(payload.get("name")),
        "birth_date": date.fromisoformat(str(raw_birth_date)),
        "period_entries": [_normalize_period_entry(entry) for entry in raw_entries],
    }


def load_local_storylines(path: Path = LOCAL_STORYLINES_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    raw_data = json.loads(path.read_text())
    raw_storylines = raw_data.get("storylines", {})
    if not isinstance(raw_storylines, Mapping):
        raise ValueError("The local story line file is malformed.")

    storylines: dict[str, dict[str, Any]] = {}
    for payload in raw_storylines.values():
        normalized = normalize_storyline_payload(payload)
        storylines[normalized["name"]] = normalized
    return _sort_storylines(storylines)


def save_local_storyline(payload: Mapping[str, Any], path: Path = LOCAL_STORYLINES_PATH) -> None:
    storylines = load_local_storylines(path)
    normalized = normalize_storyline_payload(payload)
    storylines[normalized["name"]] = normalized
    _write_local_storylines(path, storylines)


def remove_local_storyline(name: str, path: Path = LOCAL_STORYLINES_PATH) -> bool:
    storyline_name = normalize_storyline_name(name)
    storylines = load_local_storylines(path)
    removed = storylines.pop(storyline_name, None) is not None
    if not removed:
        return False
    _write_local_storylines(path, storylines)
    return True


def load_cookie_storylines(cookies: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    storylines: dict[str, dict[str, Any]] = {}
    for cookie_name, cookie_value in cookies.items():
        if not str(cookie_name).startswith(STORYLINE_COOKIE_PREFIX):
            continue
        try:
            normalized = decode_storyline_cookie_value(cookie_value)
        except ValueError:
            continue
        storylines[normalized["name"]] = normalized
    return _sort_storylines(storylines)


def storyline_cookie_name(name: object) -> str:
    storyline_name = normalize_storyline_name(name)
    digest = hashlib.sha256(storyline_name.encode("utf-8")).hexdigest()[:16]
    return f"{STORYLINE_COOKIE_PREFIX}{digest}"


def encode_storyline_cookie_value(payload: Mapping[str, Any]) -> str:
    normalized = normalize_storyline_payload(payload)
    raw_json = json.dumps(_storyline_to_json_data(normalized), separators=(",", ":"), sort_keys=True).encode("utf-8")
    compressed = zlib.compress(raw_json, level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    if len(encoded) > STORYLINE_COOKIE_MAX_VALUE_LENGTH:
        raise ValueError(
            "This story line is too large for browser-cookie storage on Streamlit Community Cloud. "
            "Try fewer saved periods or save it when running the app locally."
        )
    return encoded


def decode_storyline_cookie_value(value: str) -> dict[str, Any]:
    if not value:
        raise ValueError("Missing cookie value.")
    try:
        padding = "=" * (-len(value) % 4)
        compressed = base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))
        raw_json = zlib.decompress(compressed).decode("utf-8")
        return normalize_storyline_payload(json.loads(raw_json))
    except Exception as exc:
        raise ValueError("Invalid story line cookie value.") from exc


def build_cookie_sync_html(cookie_name: str, cookie_value: str | None) -> str:
    if cookie_value is None:
        cookie_assignment = (
            f"{cookie_name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax; Secure"
        )
    else:
        cookie_assignment = (
            f"{cookie_name}={cookie_value}; Path=/; Max-Age={STORYLINE_COOKIE_MAX_AGE_SECONDS}; "
            "SameSite=Lax; Secure"
        )

    assignment_literal = json.dumps(cookie_assignment)
    return f"""
<script>
(function () {{
  const cookieAssignment = {assignment_literal};
  const targets = [];
  try {{
    if (window.parent && window.parent.document) {{
      targets.push(window.parent.document);
    }}
  }} catch (error) {{}}
  try {{
    if (window.top && window.top.document) {{
      targets.push(window.top.document);
    }}
  }} catch (error) {{}}
  targets.push(document);
  for (const target of targets) {{
    try {{
      target.cookie = cookieAssignment;
    }} catch (error) {{
      console.warn("Unable to update cookie on target document", error);
    }}
  }}
  window.setTimeout(function () {{
    try {{
      window.top.location.reload();
    }} catch (error) {{
      try {{
        window.parent.location.reload();
      }} catch (innerError) {{
        window.location.reload();
      }}
    }}
  }}, 50);
}})();
</script>
""".strip()


def _serialize_period_entry(entry: Mapping[str, Any], *, include_boundary_geojson: bool) -> dict[str, Any]:
    serialized = {
        "place_query": str(entry.get("place_query", "") or ""),
        "resolved_name": str(entry.get("resolved_name", "") or ""),
        "latitude_text": str(entry.get("latitude_text", "") or ""),
        "longitude_text": str(entry.get("longitude_text", "") or ""),
        "coordinate_source": str(entry.get("coordinate_source", "") or ""),
        "indicator_label": str(entry.get("indicator_label", entry.get("custom_label", "")) or ""),
        "show_indicator": bool(entry.get("show_indicator", False) or str(entry.get("custom_label", "") or "").strip()),
        "bounding_box": _normalize_bounding_box(entry.get("bounding_box")),
        "end_date": _serialize_optional_date(entry.get("end_date")),
    }
    if include_boundary_geojson and entry.get("boundary_geojson") is not None:
        serialized["boundary_geojson"] = entry.get("boundary_geojson")
    return serialized


def _normalize_period_entry(entry: object) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("Each saved period entry must be a mapping.")

    return {
        "place_query": str(entry.get("place_query", "") or ""),
        "resolved_name": str(entry.get("resolved_name", "") or ""),
        "latitude_text": str(entry.get("latitude_text", "") or ""),
        "longitude_text": str(entry.get("longitude_text", "") or ""),
        "coordinate_source": str(entry.get("coordinate_source", "") or ""),
        "indicator_label": str(entry.get("indicator_label", entry.get("custom_label", "")) or ""),
        "show_indicator": bool(entry.get("show_indicator", False) or str(entry.get("custom_label", "") or "").strip()),
        "boundary_geojson": entry.get("boundary_geojson"),
        "bounding_box": _normalize_bounding_box(entry.get("bounding_box")),
        "end_date": _parse_optional_date(entry.get("end_date")),
    }


def _storyline_to_json_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": int(payload.get("version", 1)),
        "name": str(payload["name"]),
        "birth_date": payload["birth_date"].isoformat() if isinstance(payload["birth_date"], date) else str(payload["birth_date"]),
        "period_entries": [
            {
                "place_query": str(entry.get("place_query", "") or ""),
                "resolved_name": str(entry.get("resolved_name", "") or ""),
                "latitude_text": str(entry.get("latitude_text", "") or ""),
                "longitude_text": str(entry.get("longitude_text", "") or ""),
                "coordinate_source": str(entry.get("coordinate_source", "") or ""),
                "indicator_label": str(entry.get("indicator_label", "") or ""),
                "show_indicator": bool(entry.get("show_indicator", False)),
                "boundary_geojson": entry.get("boundary_geojson"),
                "bounding_box": _normalize_bounding_box(entry.get("bounding_box")),
                "end_date": _serialize_optional_date(entry.get("end_date")),
            }
            for entry in payload["period_entries"]
        ],
    }


def _write_local_storylines(path: Path, storylines: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not storylines:
        if path.exists():
            path.unlink()
        return
    json_payload = {
        "version": 1,
        "storylines": {
            name: _storyline_to_json_data(payload)
            for name, payload in _sort_storylines(dict(storylines)).items()
        },
    }
    path.write_text(json.dumps(json_payload, indent=2, sort_keys=True))


def _sort_storylines(storylines: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: dict(payload)
        for name, payload in sorted(storylines.items(), key=lambda item: item[0].casefold())
    }


def _serialize_optional_date(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)).isoformat()


def _parse_optional_date(value: object) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_bounding_box(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Bounding boxes must contain four numeric values.")
    return tuple(float(component) for component in value)
