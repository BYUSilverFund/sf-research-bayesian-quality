import datetime as dt
from pathlib import Path

import polars as pl
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
signal_name = "bayesian_barra"
signal_name_title = "Bayesian Barra"
IC = 0.05
gamma = 60
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_1")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Barra quality factor setup
quality_factors = [
    "USSLOWL_PROFIT",
    "USSLOWL_EARNQLTY",
    "USSLOWL_MGMTQLTY",
    "USSLOWL_LEVERAGE",
    "USSLOWL_GROWTH",
    "USSLOWL_EARNYILD",
]

bma_weights = {
    "USSLOWL_PROFIT": 0.001912,
    "USSLOWL_EARNQLTY": 0.000803,
    "USSLOWL_MGMTQLTY": 0.001817,
    "USSLOWL_LEVERAGE": 0.000042,
    "USSLOWL_GROWTH": -0.000323,
    "USSLOWL_EARNYILD": 0.001616,
}

# Get data
returns = sfd.load_assets(
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

factors = sfd.load_exposures(
    start=start, end=end, in_universe=True, columns=["date", "barrid"] + quality_factors
).with_columns(
    [
        (
            (pl.col(f) - pl.col(f).mean().over("date")) / pl.col(f).std().over("date")
        ).alias(f)
        for f in quality_factors
    ]
)

data = returns.join(factors, on=["date", "barrid"], how="inner")

# compute signal
signals = data.with_columns(
    pl.sum_horizontal([pl.col(f) * weight for f, weight in bma_weights.items()]).alias(
        signal_name
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
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
