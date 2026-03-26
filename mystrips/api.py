from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from mystrips.plotting import STRIPES_COLORS, export_figure_bytes, render_stripes_figure


class RenderRequest(BaseModel):
    anomalies: list[float] = Field(..., min_length=1, description="Temperature anomalies in degrees C.")
    width_px: int = Field(default=1800, ge=100, le=6000)
    height_px: int = Field(default=260, ge=20, le=2400)
    dpi: int = Field(default=200, ge=72, le=600)
    transparent_background: bool = False
    format: Literal["png", "svg", "pdf"] = "png"


app = FastAPI(
    title="MyStrips API",
    version="0.1.0",
    description=(
        "A tiny render API for MyStrips exports. Use it to automate climate-strip assets "
        "for signatures, presentations, reports, posters, or web embeds."
    ),
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "MyStrips API",
        "endpoints": ["/health", "/v1/palette", "/v1/render"],
        "suggested_uses": [
            "email signatures",
            "presentation decks",
            "reports",
            "posters",
            "web embeds",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/palette")
def palette() -> dict[str, list[str]]:
    return {"colors": STRIPES_COLORS}


@app.post("/v1/render")
def render(request: RenderRequest) -> Response:
    width_inches = request.width_px / request.dpi
    height_inches = request.height_px / request.dpi
    figure = render_stripes_figure(
        anomalies=request.anomalies,
        width_inches=width_inches,
        height_inches=height_inches,
        transparent_background=request.transparent_background,
    )
    try:
        payload = export_figure_bytes(
            figure=figure,
            fmt=request.format,
            png_dpi=request.dpi,
            transparent_background=request.transparent_background,
        )
    finally:
        plt.close(figure)

    return Response(content=payload, media_type=_media_type_for_format(request.format))


def _media_type_for_format(fmt: str) -> str:
    if fmt == "png":
        return "image/png"
    if fmt == "svg":
        return "image/svg+xml"
    return "application/pdf"
