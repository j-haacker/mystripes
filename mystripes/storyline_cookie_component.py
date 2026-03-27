from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

from mystripes.storylines import STORYLINE_COOKIE_MAX_AGE_SECONDS, STORYLINE_COOKIE_PREFIX

_COMPONENT_PATH = Path(__file__).resolve().parent / "components" / "storyline_cookie_store"
_COMPONENT = components.declare_component("storyline_cookie_store", path=str(_COMPONENT_PATH))


def sync_storyline_cookie_store(
    *,
    operation: dict[str, object] | None,
    key: str = "storyline_cookie_store",
) -> dict[str, Any] | None:
    value = _COMPONENT(
        cookie_prefix=STORYLINE_COOKIE_PREFIX,
        max_age_seconds=STORYLINE_COOKIE_MAX_AGE_SECONDS,
        operation=operation,
        default=None,
        key=key,
    )
    if not isinstance(value, dict):
        return None
    return dict(value)
