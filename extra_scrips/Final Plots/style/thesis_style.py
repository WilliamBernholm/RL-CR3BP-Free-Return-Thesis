"""
Unified thesis plotting style for all final figures.

Use:
    from style.thesis_style import apply_thesis_style, get_figsize, save_thesis_figure

    apply_thesis_style()
    fig, ax = plt.subplots(figsize=get_figsize("single"))
"""

from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# FIGURE SIZES
# ============================================================



# Good for one-column figures
FIGSIZE_SINGLE = (5.2, 3.2)

# Good for two-column wide figures
FIGSIZE_DOUBLE = (10.8, 4.2)

# Optional taller version for heatmaps or trajectory panels
FIGSIZE_DOUBLE_TALL = (10.8, 5.4)


# ============================================================
# FONT SIZES
# ============================================================

FONT_FAMILY = "DejaVu Sans"

TITLE_SIZE = 11
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8
COLORBAR_LABEL_SIZE = 10
COLORBAR_TICK_SIZE = 9


# ============================================================
# LINE / MARKER STYLE
# ============================================================

LINEWIDTH_MAIN = 1.8
LINEWIDTH_SECONDARY = 1.2
LINEWIDTH_THIN = 0.8

MARKER_SIZE = 4
GRID_LINEWIDTH = 0.5
AXIS_LINEWIDTH = 0.8


# ============================================================
# EXPORT SETTINGS
# ============================================================

DPI_PNG = 600
DPI_PREVIEW = 200

SAVE_PDF = False
SAVE_PNG = True

BBOX = "tight"
PAD_INCHES = 0.03



IEEE_LOWERCASE_WORDS = {
    "a", "an", "and", "as", "at", "but",
    "by", "for", "from", "in", "into",
    "nor", "of", "on", "onto", "or",
    "over", "per", "the", "to", "up",
    "via", "with"
}


def ieee_title(text: str) -> str:
    """
    Convert title to IEEE-style capitalization
    while preserving acronyms like PPO-MCC, TLI, CR3BP, etc.
    """

    words = text.split()

    result = []

    for i, word in enumerate(words):

        lower = word.lower()

        # Preserve acronyms / all-caps tokens
        if any(c.isupper() for c in word):
            result.append(word)
            continue

        # IEEE lowercase connector words
        if i != 0 and lower in IEEE_LOWERCASE_WORDS:
            result.append(lower)
            continue

        # Handle hyphenated words
        if "-" in word:

            pieces = word.split("-")

            pieces = [
                p.capitalize() if p.lower() not in IEEE_LOWERCASE_WORDS else p.lower()
                for p in pieces
            ]

            result.append("-".join(pieces))

        else:
            result.append(word.capitalize())

    return " ".join(result)


def apply_thesis_style():
    """Apply global Matplotlib style for thesis-ready figures."""

    plt.rcParams.update({
        "font.family": FONT_FAMILY,

        "figure.dpi": DPI_PREVIEW,
        "savefig.dpi": DPI_PNG,

        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,

        "axes.linewidth": AXIS_LINEWIDTH,

        "lines.linewidth": LINEWIDTH_MAIN,
        "lines.markersize": MARKER_SIZE,

        "grid.linewidth": GRID_LINEWIDTH,
        "grid.alpha": 0.35,

        "legend.frameon": True,
        "legend.framealpha": 0.90,

        "figure.autolayout": False,

        "mathtext.fontset": "dejavuserif",

        
    })


def get_figsize(kind="single"):
    """
    Return standard thesis figure size.

    Options:
        "single"       one-column figure
        "double"       two-column wide figure
        "double_tall"  two-column figure with more vertical space
    """

    kind = str(kind).lower()

    if kind in ("single", "one_column", "small"):
        return FIGSIZE_SINGLE

    if kind in ("double", "two_column", "wide"):
        return FIGSIZE_DOUBLE

    if kind in ("double_tall", "wide_tall", "heatmap"):
        return FIGSIZE_DOUBLE_TALL

    raise ValueError(f"Unknown figure size kind: {kind}")


def clean_axis(ax, grid=True):
    """Apply common axis formatting."""

    if grid:
        ax.grid(True)

    ax.tick_params(direction="in", top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)


def save_thesis_figure(fig, output_path, save_pdf=SAVE_PDF, save_png=SAVE_PNG):
    """
    Save a figure as thesis-ready PNG and/or PDF.

    Example:
        save_thesis_figure(fig, Path("outputs/thesis_ready/reward_plot"))
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = output_path.with_suffix("")

    if save_png:
        fig.savefig(
            stem.with_suffix(".png"),
            dpi=DPI_PNG,
            bbox_inches=BBOX,
            pad_inches=PAD_INCHES,
        )

    if save_pdf:
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches=BBOX,
            pad_inches=PAD_INCHES,
        )


def add_panel_label(ax, label, x=0.02, y=0.96):
    """Add small subplot label, for example (a), (b), etc."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )