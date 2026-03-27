from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

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
