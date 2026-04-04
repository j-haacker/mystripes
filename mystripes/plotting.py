from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

STRIPES_COLORS = [
    "#08306b",
    "#08519c",
    "#2171b5",
    "#4292c6",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#deebf7",
    "#fee0d2",
    "#fcbba1",
    "#fc9272",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#a50f15",
    "#67000d",
]

_WATERMARK_PADDING_RATIO = 0.02
_PERIOD_INDICATOR_MIN_GAP_PIXELS = 1.0

_WATERMARK_HORIZONTAL_POSITIONS = {
    "left": (_WATERMARK_PADDING_RATIO, "left", 1 - _WATERMARK_PADDING_RATIO),
    "center": (0.5, "center", 1 - (2 * _WATERMARK_PADDING_RATIO)),
    "right": (1 - _WATERMARK_PADDING_RATIO, "right", 1 - _WATERMARK_PADDING_RATIO),
}
_WATERMARK_VERTICAL_POSITIONS = {
    "bottom": (_WATERMARK_PADDING_RATIO, "bottom", 1 - _WATERMARK_PADDING_RATIO),
    "center": (0.5, "center", 1 - (2 * _WATERMARK_PADDING_RATIO)),
    "top": (1 - _WATERMARK_PADDING_RATIO, "top", 1 - _WATERMARK_PADDING_RATIO),
}


def render_stripes_figure(
    anomalies: list[float],
    width_inches: float,
    height_inches: float,
    watermark_text: str | None = None,
    watermark_horizontal_align: str = "center",
    watermark_vertical_align: str = "center",
    watermark_color: str = "#ffffff",
    watermark_opacity: float = 0.35,
    watermark_max_width_ratio: float = 0.8,
    watermark_max_height_ratio: float = 0.8,
    period_indicators: Sequence[Mapping[str, object]] | None = None,
    period_indicator_style: str = "scale_bar",
    period_indicator_vertical_align: str = "bottom",
    period_indicator_color: str = "#ffffff",
) -> plt.Figure:
    if not anomalies:
        raise ValueError("At least one anomaly is required to render strips.")

    values = np.array([anomalies], dtype=float)
    color_limit = max(float(np.max(np.abs(values))), 0.25)

    figure, axis = plt.subplots(
        figsize=(width_inches, height_inches),
        dpi=100,
    )
    axis.imshow(
        values,
        aspect="auto",
        cmap=ListedColormap(STRIPES_COLORS),
        interpolation="nearest",
        vmin=-color_limit,
        vmax=color_limit,
    )
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _add_watermark(
        axis=axis,
        text=watermark_text,
        horizontal_align=watermark_horizontal_align,
        vertical_align=watermark_vertical_align,
        color=watermark_color,
        opacity=watermark_opacity,
        max_width_ratio=watermark_max_width_ratio,
        max_height_ratio=watermark_max_height_ratio,
    )
    _add_period_indicators(
        axis=axis,
        indicators=period_indicators,
        style=period_indicator_style,
        vertical_align=period_indicator_vertical_align,
        color=period_indicator_color,
    )
    return figure


def export_figure_bytes(
    figure: plt.Figure,
    fmt: str,
    png_dpi: int,
) -> bytes:
    buffer = BytesIO()
    save_kwargs: dict[str, object] = {"format": fmt}
    if fmt == "png":
        save_kwargs["dpi"] = png_dpi
    figure.savefig(buffer, **save_kwargs)
    return buffer.getvalue()


def _add_watermark(
    axis,
    text: str | None,
    horizontal_align: str,
    vertical_align: str,
    color: str,
    opacity: float,
    max_width_ratio: float,
    max_height_ratio: float,
) -> None:
    content = str(text or "").strip()
    if not content:
        return
    if not 0 <= opacity <= 1:
        raise ValueError("The watermark opacity must be between 0 and 1.")
    if not 0 < max_width_ratio <= 1:
        raise ValueError("The watermark maximum width ratio must be between 0 and 1.")
    if not 0 < max_height_ratio <= 1:
        raise ValueError("The watermark maximum height ratio must be between 0 and 1.")
    if horizontal_align not in _WATERMARK_HORIZONTAL_POSITIONS:
        raise ValueError(f"Unsupported watermark horizontal alignment: {horizontal_align}")
    if vertical_align not in _WATERMARK_VERTICAL_POSITIONS:
        raise ValueError(f"Unsupported watermark vertical alignment: {vertical_align}")

    x_position, text_horizontal_align, available_width_ratio = _WATERMARK_HORIZONTAL_POSITIONS[horizontal_align]
    y_position, text_vertical_align, available_height_ratio = _WATERMARK_VERTICAL_POSITIONS[vertical_align]
    text_artist = axis.text(
        x_position,
        y_position,
        content,
        transform=axis.transAxes,
        ha=text_horizontal_align,
        va=text_vertical_align,
        color=color,
        alpha=opacity,
        clip_on=True,
        fontsize=100,
        fontweight="bold",
        zorder=3,
    )

    figure = axis.figure
    for _ in range(2):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        axis_box = axis.get_window_extent(renderer=renderer)
        text_box = text_artist.get_window_extent(renderer=renderer)
        if text_box.width <= 0 or text_box.height <= 0:
            break

        max_width_pixels = axis_box.width * min(max_width_ratio, available_width_ratio)
        max_height_pixels = axis_box.height * min(max_height_ratio, available_height_ratio)
        width_scale = max_width_pixels / text_box.width
        height_scale = max_height_pixels / text_box.height
        target_scale = min(width_scale, height_scale) * 0.98
        text_artist.set_fontsize(max(float(text_artist.get_fontsize()) * target_scale, 1.0))


def _add_period_indicators(
    axis,
    indicators: Sequence[Mapping[str, object]] | None,
    style: str,
    vertical_align: str,
    color: str,
) -> None:
    if not indicators:
        return
    if style not in {"scale_bar", "outward_arrows"}:
        raise ValueError(f"Unsupported period indicator style: {style}")
    if vertical_align not in {"top", "bottom"}:
        raise ValueError(f"Unsupported period indicator vertical alignment: {vertical_align}")

    normalized_indicators = []
    for indicator in indicators:
        if not isinstance(indicator, Mapping):
            raise ValueError("Each period indicator must be a mapping.")
        label = str(indicator.get("label", "") or "").strip()
        start_fraction = float(indicator.get("start_fraction", 0.0) or 0.0)
        end_fraction = float(indicator.get("end_fraction", 0.0) or 0.0)
        start_fraction = max(0.0, min(1.0, start_fraction))
        end_fraction = max(0.0, min(1.0, end_fraction))
        if end_fraction <= start_fraction:
            continue
        normalized_indicators.append(
            {
                "label": label,
                "start_fraction": start_fraction,
                "end_fraction": end_fraction,
            }
        )

    if not normalized_indicators:
        return

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_box = axis.get_window_extent(renderer=renderer)
    if axis_box.width <= 0 or axis_box.height <= 0:
        return

    gap_ratio = _PERIOD_INDICATOR_MIN_GAP_PIXELS / axis_box.width
    adjusted_indicators = _apply_indicator_gap(normalized_indicators, gap_ratio)
    line_y = 0.12 if vertical_align == "bottom" else 0.88
    label_y = 0.03 if vertical_align == "bottom" else 0.97
    label_vertical_align = "bottom" if vertical_align == "top" else "top"
    line_width = max(1.0, min(2.2, axis_box.height / 140.0))
    tick_half_height = max(0.012, min(0.04, 8.0 / axis_box.height))
    arrow_scale = max(8.0, min(16.0, axis_box.height / 9.0))
    base_fontsize = max(7.0, min(13.0, axis_box.height / 18.0))
    text_artists = []

    for indicator in adjusted_indicators:
        start_fraction = indicator["start_fraction"]
        end_fraction = indicator["end_fraction"]
        if style == "scale_bar":
            axis.add_line(
                Line2D(
                    [start_fraction, end_fraction],
                    [line_y, line_y],
                    transform=axis.transAxes,
                    color=color,
                    linewidth=line_width,
                    solid_capstyle="butt",
                    alpha=0.92,
                    zorder=4,
                )
            )
            for x_position in (start_fraction, end_fraction):
                axis.add_line(
                    Line2D(
                        [x_position, x_position],
                        [line_y - tick_half_height, line_y + tick_half_height],
                        transform=axis.transAxes,
                        color=color,
                        linewidth=line_width,
                        solid_capstyle="butt",
                        alpha=0.92,
                        zorder=4,
                    )
                )
        else:
            axis.add_patch(
                FancyArrowPatch(
                    (start_fraction, line_y),
                    (end_fraction, line_y),
                    transform=axis.transAxes,
                    arrowstyle="<|-|>",
                    mutation_scale=arrow_scale,
                    linewidth=line_width,
                    color=color,
                    alpha=0.92,
                    zorder=4,
                    clip_on=True,
                    shrinkA=0,
                    shrinkB=0,
                )
            )

        label = str(indicator["label"])
        if label:
            text_artists.append(
                (
                    axis.text(
                        (start_fraction + end_fraction) / 2.0,
                        label_y,
                        label,
                        transform=axis.transAxes,
                        ha="center",
                        va=label_vertical_align,
                        color=color,
                        clip_on=True,
                        fontsize=base_fontsize,
                        fontweight="bold",
                        zorder=5,
                        path_effects=[
                            patheffects.Stroke(linewidth=2.2, foreground=(0.0, 0.0, 0.0, 0.5)),
                            patheffects.Normal(),
                        ],
                    ),
                    end_fraction - start_fraction,
                )
            )

    if not text_artists:
        return

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text_artist, width_fraction in text_artists:
        text_box = text_artist.get_window_extent(renderer=renderer)
        if text_box.width <= 0 or width_fraction <= 0:
            continue
        available_width_pixels = axis_box.width * width_fraction
        if available_width_pixels <= 0:
            continue
        if text_box.width <= available_width_pixels:
            continue
        scale = max(0.55, min(1.0, (available_width_pixels / text_box.width) * 0.95))
        text_artist.set_fontsize(max(float(text_artist.get_fontsize()) * scale, 5.5))


def _apply_indicator_gap(
    indicators: list[dict[str, object]],
    gap_ratio: float,
) -> list[dict[str, object]]:
    adjusted = [dict(indicator) for indicator in sorted(indicators, key=lambda item: float(item["start_fraction"]))]
    if not adjusted:
        return adjusted

    half_gap = gap_ratio / 2.0
    for index in range(len(adjusted) - 1):
        current = adjusted[index]
        following = adjusted[index + 1]
        current_start = float(current["start_fraction"])
        current_end = float(current["end_fraction"])
        next_start = float(following["start_fraction"])
        next_end = float(following["end_fraction"])
        minimum_next_start = current_end + gap_ratio
        if next_start >= minimum_next_start:
            continue
        midpoint = (current_end + next_start) / 2.0
        new_current_end = max(current_start, midpoint - half_gap)
        new_next_start = min(next_end, midpoint + half_gap)
        current["end_fraction"] = max(current_start, new_current_end)
        following["start_fraction"] = min(max(new_next_start, current["end_fraction"]), next_end)
    return adjusted
