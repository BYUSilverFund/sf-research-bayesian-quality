# Quality Only

import datetime as dt
from pathlib import Path

import great_tables as gt
import matplotlib.pyplot as plt
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
import sf_quant.research as sfr

from research.utils import run_backtest_parallel

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "qarp"
IC = 0.05
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/brandon/experiment_2")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Get data
data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "daily_volume",
        "market_cap",
        "specific_return",
        "specific_risk",
        "predicted_beta",
        "bid_ask_spread",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

exposures = sfd.load_exposures(
    start=start,
    end=end,
    in_universe=True,
    columns=[
        "date",
        "barrid",
        "USSLOWL_EARNQLTY",
        "USSLOWL_VALUE",
        "USSLOWL_PROFIT",
    ],
)


data = data.join(exposures, on=["date", "barrid"], how="left")

signals = data.sort("barrid", "date").with_columns(
    (
        (pl.col("USSLOWL_EARNQLTY").add(pl.col("USSLOWL_PROFIT")).truediv(2)).add(
            pl.col("USSLOWL_VALUE")
        )
    )
    .truediv(2)
    .shift(2)
    .over("barrid")
    .alias(signal_name)
)


# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col(signal_name).is_not_nan(),
)

signal_stats = sfr.get_signal_stats(filtered, column=signal_name)

signal_stats_table = (
    gt.GT(signal_stats)
    .tab_header(title=f"{signal_name} Summary Statistics")
    .cols_label(
        mean="Mean",
        std="Std.",
        min="Minimum",
        q25="25th Percentile",
        q50="50th Percentile",
        q75="75th Percentile",
        max="Maximum",
    )
    # .fmt_percent(["mean_return", "volatility"], decimals=2)
    .fmt_number(["mean", "std", "min", "max", "q25", "q50", "q75"], decimals=2)
    .opt_stylize(style=4, color="gray")
)

signal_stats_table_path = results_folder / f"{signal_name}_stats.png"
signal_stats_table.save(signal_stats_table_path, scale=3)

signal_values = filtered.select(signal_name).to_numpy().flatten()
distribution_chart_path = results_folder / f"{signal_name}_distribution.png"

plt.figure(figsize=(10, 6))
plt.hist(signal_values, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
# plt.xlim(0, 10)
plt.title(f"{signal_name} Distribution")
plt.xlabel(f"{signal_name} Value")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(distribution_chart_path)


# Compute scores
scores = filtered.select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

# Get forward returns
# forward_returns = (
#     data.sort("date", "barrid")
#     .select(
#         "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("fwd_return")
#     )
#     .drop_nulls("fwd_return")
# )

returns = data.sort("date", "barrid").select("date", "barrid", "return")

# Merge alphas and returns
# merged = alphas.join(other=returns, on=["date", "barrid"], how="inner")

# # Get merged alphas and forward returns (inner join)
# merged_alphas = merged.select("date", "barrid", "alpha")
# # merged_forward_returns = merged.select("date", "barrid", "fwd_return")
# merged_forward_returns = merged.select("date", "barrid", "return", "fwd_return")


# Get ics
ics = sfp.generate_alpha_ics(alphas=alphas, rets=returns, method="rank", window=22)

# Save ic chart
rank_chart_path = results_folder / "rank_ic_chart.png"
pearson_chart_path = results_folder / "pearson_ic_chart.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)
# sfp.generate_ic_chart(
#     ics=ics,
#     title=f"{signal_name} Cumulative IC",
#     ic_type="Pearson",
#     file_name=pearson_chart_path,
# )

# Run parallelized backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
