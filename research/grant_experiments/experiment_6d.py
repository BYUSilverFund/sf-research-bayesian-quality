from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

# Parameters
checkpoint_dir = "temp/checkpoints_bma_factor_60"
results_folder = Path("results/experiment_6")
results_folder.mkdir(parents=True, exist_ok=True)


def plot_faceted_timeseries():
    print(f"Loading checkpoint data from {checkpoint_dir}...")

    # Read and combine all parquet files
    try:
        df = pl.scan_parquet(f"{checkpoint_dir}/*.parquet").collect().sort("date")
    except Exception as e:
        print(f"Error loading parquets: {e}")
        return

    # Convert to pandas and ensure date is proper datetime
    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])

    # Extract the factor names
    factors = [col for col in pdf.columns if col != "date"]
    n_factors = len(factors)

    # Create the Faceted Plot
    sns.set_theme(style="whitegrid")

    # Create subplots: 1 column, multiple rows. sharex=True perfectly aligns the dates.
    fig, axes = plt.subplots(
        nrows=n_factors, ncols=1, figsize=(12, 2 * n_factors), sharex=True
    )

    colors = sns.color_palette("husl", n_factors)

    # Plot each factor on its own dedicated axis
    for i, factor in enumerate(factors):
        ax = axes[i]

        # Plot the line
        ax.plot(pdf["date"], pdf[factor], color=colors[i], linewidth=1.5, alpha=0.9)

        # Add the zero-line reference
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, zorder=0)

        # Formatting for the individual subplot
        ax.set_title(
            factor, loc="left", fontsize=11, fontweight="bold", color=colors[i]
        )
        ax.set_ylabel("Beta", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    # Global Formatting
    plt.xlabel("Date", fontsize=12, fontweight="bold")
    fig.suptitle("Rolling BMA Expected Betas", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    # Save and show
    save_path = results_folder / "rolling_betas_faceted.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved successfully to {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_faceted_timeseries()
