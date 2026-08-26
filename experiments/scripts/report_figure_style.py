"""Shared publication style and atomic output helpers for experiment figures."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


NAVY = "#1F4E79"
ORANGE = "#E67E22"
TEAL = "#009E73"
VERMILION = "#D55E00"
INK = "#20262E"
MUTED = "#5D6875"
GRID = "#D9E0E7"
PALE = "#F4F6F8"
NEUTRAL = "#8A96A3"
WHITE = "#FFFFFF"


class FigureBuildError(ValueError):
    """Raised when a publication figure package cannot be built safely."""


def require(condition, message):
    if not condition:
        raise FigureBuildError(message)


def style_context(hash_salt="warehouse-object-detection-report-figures-v1"):
    """Return the shared, deterministic Matplotlib style context."""

    return plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.frameon": False,
            "svg.hashsalt": hash_salt,
        }
    )


def add_header(fig, title, subtitle, *, title_x=0.06):
    """Add the shared title and explanatory subtitle to a figure."""

    fig.text(
        title_x,
        0.965,
        str(title),
        fontsize=19,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        title_x,
        0.915,
        str(subtitle),
        fontsize=10.5,
        color=MUTED,
        va="top",
    )


def clean_axis(axis, grid_axis="x", *, keep_left=False):
    """Apply restrained axes and neutral grid lines."""

    hidden = ["top", "right"] if keep_left else ["top", "right", "left"]
    axis.spines[hidden].set_visible(False)
    if grid_axis:
        axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)


def three_panel_figure(title, subtitle, panels):
    """Build the approved three-panel input/control/decision visual language."""

    require(len(panels) == 3, "The approved structural visual requires three panels.")
    accents = (INK, MUTED, TEAL)
    fig = plt.figure(figsize=(13.6, 5.7))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    add_header(fig, title, subtitle)

    lefts = (0.055, 0.37, 0.685)
    for left, panel, default_accent in zip(lefts, panels, accents):
        heading = str(panel["heading"]).upper()
        bullets = [str(value) for value in panel["bullets"]]
        require(1 <= len(bullets) <= 5, f"{heading} must contain one to five bullets.")
        accent = str(panel.get("accent", default_accent))
        card = FancyBboxPatch(
            (left, 0.12),
            0.27,
            0.65,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=WHITE,
            edgecolor=GRID,
            linewidth=1.2,
            transform=axis.transAxes,
        )
        axis.add_patch(card)
        axis.add_patch(
            Rectangle(
                (left, 0.70),
                0.27,
                0.07,
                transform=axis.transAxes,
                facecolor=accent,
                edgecolor="none",
            )
        )
        axis.text(
            left + 0.018,
            0.735,
            heading,
            transform=axis.transAxes,
            color=WHITE,
            fontsize=12,
            fontweight="bold",
            va="center",
        )

        y = 0.645
        base_step = 0.49 / max(len(bullets) - 1, 1)
        for bullet in bullets:
            wrapped = textwrap.fill(bullet, width=35)
            axis.text(
                left + 0.022,
                y,
                "•",
                transform=axis.transAxes,
                color=accent,
                fontsize=14,
                va="top",
            )
            axis.text(
                left + 0.044,
                y,
                wrapped,
                transform=axis.transAxes,
                color=INK,
                fontsize=10.3,
                va="top",
                linespacing=1.25,
            )
            y -= min(base_step, 0.16)
    return fig


def save_figure_png(fig, directory, stem):
    """Write one deterministic PNG and close the figure."""

    target = Path(directory)
    png = target / f"{stem}.png"
    fig.savefig(
        png,
        dpi=200,
        bbox_inches="tight",
        facecolor=WHITE,
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)
    require(png.is_file() and png.stat().st_size > 0, f"PNG was not written: {png}")


def save_figure_pair(fig, directory, stem):
    """Write a deterministic PNG/SVG pair and close the figure."""

    target = Path(directory)
    png = target / f"{stem}.png"
    svg = target / f"{stem}.svg"
    fig.savefig(
        png,
        dpi=200,
        bbox_inches="tight",
        facecolor=WHITE,
        metadata={"Software": "Matplotlib; verified experiment figure builder"},
    )
    fig.savefig(
        svg,
        format="svg",
        bbox_inches="tight",
        facecolor=WHITE,
        metadata={"Date": None, "Creator": "Verified experiment figure builder"},
    )
    plt.close(fig)
    # Normalize generated text so SVG bytes are stable and pass repository
    # whitespace checks on Windows, WSL, and CI.
    content = svg.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    content = b"\n".join(line.rstrip(b" \t") for line in content.split(b"\n"))
    svg.write_bytes(content)
    require(png.is_file() and png.stat().st_size > 0, f"PNG was not written: {png}")
    require(svg.is_file() and svg.stat().st_size > 0, f"SVG was not written: {svg}")


def build_atomic_package(output_dir, builders, *, hash_salt):
    """Render named figure builders and atomically promote their directory."""

    destination = Path(output_dir).expanduser().absolute()
    require(not destination.exists(), f"Refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.incomplete"
    require(not staging.exists(), f"Incomplete figure build already exists: {staging}")
    staging.mkdir()
    stems = [str(stem) for stem, _ in builders]
    require(len(stems) == len(set(stems)), "Figure stems must be unique.")

    try:
        with style_context(hash_salt):
            for stem, builder in builders:
                save_figure_pair(builder(), staging, stem)
        expected = {f"{stem}.{extension}" for stem in stems for extension in ("png", "svg")}
        actual = {path.name for path in staging.iterdir() if path.is_file()}
        require(actual == expected, "Rendered figure package has an unexpected file set.")
        os.replace(staging, destination)
    except Exception:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
        raise
    return destination
