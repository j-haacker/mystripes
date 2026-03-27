from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_COMPONENT_PATH = Path(__file__).resolve().parent / "components" / "storyline_cookie_store"
_COMPONENT = components.declare_component("storyline_cookie_store", path=str(_COMPONENT_PATH))


def sync_storyline_cookie_store(
    *,
    monitored_cookie_names: list[str],
    monitored_cookie_prefixes: list[str],
    operation: dict[str, object] | None,
    key: str = "storyline_cookie_store",
) -> dict[str, Any] | None:
    value = _COMPONENT(
        cookie_names=monitored_cookie_names,
        cookie_prefixes=monitored_cookie_prefixes,
        operation=operation,
        default=None,
        key=key,
    )
    if not isinstance(value, dict):
        return None
    return dict(value)
