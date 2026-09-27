"""Figures derived from the same recorded numbers as the tables."""

from pathlib import Path
from typing import Any


def reliability_diagram(table: list[dict[str, Any]], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in table if r["count"]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.bar(
        [r["low"] for r in table],
        [r["accuracy"] or 0.0 for r in table],
        width=[r["high"] - r["low"] for r in table],
        align="edge",
        edgecolor="black",
        alpha=0.6,
    )
    ax.plot([r["mean_confidence"] for r in rows], [r["accuracy"] for r in rows], "o", color="C1")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="top-label confidence", ylabel="accuracy", title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
