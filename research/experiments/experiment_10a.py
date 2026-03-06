import datetime as dt
import os
from pathlib import Path

import polars as pl
import polars_ols as pls  # noqa
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv
from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

# Load environment variables
load_dotenv(override=True)

# Parameters
start = dt.date(1996, 6, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "vol_scaled_capm_momentum"
signal_name_title = "Vol. Scaled CAPM Idio. Momentum"
IC = 0.05
gamma = 60
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_10")
project_root = os.getenv("PROJECT_ROOT")
byu_email = os.getenv("BYU_EMAIL")

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

# Compute capm betas
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

# Compute signal
signals = (
    capm_betas.sort("barrid", "date")
    .with_columns(
        pl.col("return_rf")
        .sub(
            pl.sum_horizontal(
                pl.col("B_const"),
                pl.col("B_mkt_rf").mul("mkt_rf"),
            )
        )
        .alias("residual"),
    )
    .with_columns(
        pl.col("residual")
        .log1p()
        .rolling_sum(230)
        .truediv(pl.col("residual").rolling_std(230))
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

alphas.write_parquet(f"{signal_name}_alphas.parquet")

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

# Get merged alphas and forward returns (inner join)
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
slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

backtest_config = BacktestConfig(
    signal_name=signal_name,
    data_path=f"{signal_name}_alphas.parquet",
    gamma=gamma,
    project_root=project_root,
    byu_email=byu_email,
    constraints=constraints,
    slurm=slurm_config,
)

backtest_runner = BacktestRunner(backtest_config)
backtest_runner.submit()
