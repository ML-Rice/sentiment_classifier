from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "Amazon_Reviews_clean.csv"


def plot_helpfulness_analysis(csv_path: Path):
    # 1. Load data from disk (read-only)
    df = pd.read_csv(csv_path)

    # 2. Filter out reviews with 0 votes to prevent division by zero (0 / 0)
    voted_df = df[df["HelpfulnessDenominator"] > 0].copy()

    # 3. Calculate helpfulness ratio dynamically in memory
    voted_df["helpful_ratio"] = (
        voted_df["HelpfulnessNumerator"] / voted_df["HelpfulnessDenominator"]
    )

    # =============================================================
    # WINDOW 1: Standard Scatter Plot
    # =============================================================
    plt.figure(figsize=(7, 6))

    plt.scatter(
        voted_df["Score"],
        voted_df["helpful_ratio"],
        facecolors="none",      # Makes the interior of the circles hollow
        edgecolors="#1f77b4",   # Colors the outline of each circle
        s=15,                   # Smaller marker size to reduce clutter
        alpha=0.3,              # Transparency shows density where rings stack
        linewidths=0.7          # Thin stroke for crisp outlines
    )
    plt.title(
        "Raw Helpfulness Ratio Distribution (Scatter)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Review Score (1-5 Stars)")
    plt.ylabel("Helpfulness Ratio")
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(-0.05, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # =============================================================
    # WINDOW 2: Bar Chart (Averages)
    # =============================================================
    plt.figure(figsize=(7, 6))

    grouped_data = voted_df.groupby("Score")["helpful_ratio"].mean()
    scores = grouped_data.index
    means = grouped_data.values

    bars = plt.bar(scores, means, color="#1f77b4", edgecolor="black", alpha=0.8)

    plt.title(
        "Mean Helpfulness Ratio by Rating", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Review Score (1-5 Stars)")
    plt.ylabel("Average Helpfulness Ratio")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add numeric labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # =============================================================
    # WINDOWS 3-7: Individual Distribution Window for Each Score (1 to 5)
    # =============================================================
    for score in range(1, 6):
        # Create a new independent figure window for each score
        plt.figure(figsize=(6, 4.5))

        score_ratios = voted_df[voted_df["Score"] == score]["helpful_ratio"]

        plt.hist(
            score_ratios,
            bins=10,
            range=(0, 1),
            color="#1f77b4",
            edgecolor="black",
            alpha=0.7,
        )

        plt.title(
            f"Helpfulness Ratio Distribution: {score}-Star Reviews",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Helpfulness Ratio (0.0 to 1.0)")
        plt.ylabel("Number of Reviews")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
    # Open windows all at once

    plt.show()


if __name__ == "__main__":
    plot_helpfulness_analysis(PROCESSED_PATH)