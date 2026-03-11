from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Parameters
checkpoint_dir = "temp/checkpoints_bma_60m"
results_folder = Path("results/experiment_3")
results_folder.mkdir(parents=True, exist_ok=True)


def plot_distributions():
    print(f"Loading checkpoint data from {checkpoint_dir}...")

    # Read and combine all parquet files in the folder
    try:
        df = pl.scan_parquet(f"{checkpoint_dir}/*.parquet").collect().sort("date")
    except Exception as e:
        print(f"Error loading parquets. Make sure the folder exists and has files: {e}")
        return

    # Melt the dataframe for plotting
    melted_df = df.unpivot(
        index="date", variable_name="factor", value_name="beta"
    ).to_pandas()

    # Create the Distribution Plot
    plt.figure(figsize=(12, 7))

    # Create a boxplot to show the interquartile ranges and median
    sns.boxplot(
        data=melted_df, x="factor", y="beta", color="whitesmoke", showfliers=False
    )

    # Overlay a stripplot to show the actual density of the monthly roll values
    sns.stripplot(
        data=melted_df,
        x="factor",
        y="beta",
        alpha=0.4,
        jitter=True,
        size=4,
        palette="viridis",
        hue="factor",
        legend=False,
    )

    # Formatting
    plt.title(
        "Distribution of Rolling BMA Expected Betas (60-Month Window)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Barra Quality Factor", fontsize=12)
    plt.ylabel("Probability-Weighted Beta", fontsize=12)
    plt.xticks(rotation=25, ha="right")

    # Add a zero-line to easily see when factors flipped negative
    plt.axhline(0, color="red", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()

    # Save and show
    save_path = results_folder / "rolling_betas_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_distributions()
