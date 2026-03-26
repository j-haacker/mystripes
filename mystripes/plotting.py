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


def render_stripes_figure(
    anomalies: list[float],
    width_inches: float,
    height_inches: float,
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
