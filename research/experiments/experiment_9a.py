import datetime as dt
from pathlib import Path

import polars as pl
import polars_ols as pls  # noqa
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "capm_momentum"
signal_name_title = "CAPM Momentum"
IC = 0.05
gamma = 60
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_9")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load data
assets = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "specific_return",
        "specific_risk",
        "predicted_beta",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# Get fama french data for mkt_rf and rf
ff = sfd.load_fama_french(start=start, end=end).select("date", "mkt_rf", "rf")

# Join ff and sfq data
data = assets.join(ff, on="date", how="left")

# Run CAPM regression
capm_betas = (
    data.sort("barrid", "date")
    .with_columns(
        pl.col("return").sub("rf").alias("return_rf"), pl.lit(1.0).alias("const")
    )
    .with_columns(
        pl.col("return_rf")
        .least_squares.rolling_ols(
            pl.col("const", "mkt_rf"),
            window_size=252,
            min_periods=252,
            mode="coefficients",
        )
        .over("barrid")
        .alias("B")
    )
    .unnest("B", separator="_")
)

# Momentum signal
signals = (
    capm_betas.sort("barrid", "date")
    .with_columns(
        pl.col("return_rf")
        .sub(
            pl.sum_horizontal(
                pl.col("B_const"), pl.col("B_mkt_rf").mul(pl.col("mkt_rf"))
            )
        )
        .alias("residual")
    )
    .with_columns(
        pl.col("residual")
        .log1p()
        .rolling_sum(232)
        .shift(21)
        .over("barrid")
        .alias(signal_name)
    )
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

# Compute z scores
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
forward_returns = (
    data.sort("date", "barrid")
    .select(
        "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("fwd_return")
    )
    .drop_nulls("fwd_return")
)

# Merge alphas and forward returns
merged = alphas.join(other=forward_returns, on=["date", "barrid"], how="inner")

# Get merged alphas and forward returns
merged_alphas = merged.select("date", "barrid", "alpha")
merged_forward_returns = merged.select("date", "barrid", "fwd_return")

# Get ics
ics = sfp.generate_alpha_ics(
    alphas=alphas, rets=forward_returns, method="rank", window=22
)

# Save ic chart
rank_chart_path = results_folder / "rank_ic_chart.png"
pearson_chart_path = results_folder / "pearson_ic_chart.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name_title} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name_title} Cumulative IC",
    ic_type="Pearson",
    file_name=pearson_chart_path,
)

# Run parallelized backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
