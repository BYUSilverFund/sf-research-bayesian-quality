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
signal_name = "ols_qmj"
IC = 0.05
gamma = 30
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/josh/experiment_3")

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

quality_cols = [
    "USSLOWL_PROFIT",
    "USSLOWL_EARNQLTY",
    "USSLOWL_MGMTQLTY",
    "USSLOWL_LEVERAGE",
    "USSLOWL_GROWTH",
]

signal_weights = {
    "USSLOWL_PROFIT": 0.5571,
    "USSLOWL_EARNQLTY": 0.4460,
    "USSLOWL_MGMTQLTY": 0.0238,
    "USSLOWL_LEVERAGE": -0.0177,
    "USSLOWL_GROWTH": -0.0092
}

factors = sfd.load_exposures(
    start=start, end=end, in_universe=True, columns=["date", "barrid"] + quality_cols
).with_columns(
    [
        (
            (pl.col(f) - pl.col(f).mean().over("date")) / pl.col(f).std().over("date")
        ).alias(f)
        for f in quality_cols
    ]
)

data = data.join(exposures, on=["date", "barrid"], how="left")

data = data.join(factors, on=["date", "barrid"], how="inner")

# static QMJ signal
signals = (
    data
    .sort("barrid", "date")
    .with_columns(
        sum(
            pl.col(factor) * signal_weights[factor]
            for factor in quality_cols
        )
        .shift(2)
        .over('barrid')
        .alias(signal_name)
    )
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col(signal_name).is_not_nan(),
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
        "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("return")
    )
    .drop_nulls("return")
)

# Get ics
ics = sfp.generate_alpha_ics(alphas=alphas, rets=forward_returns, method="rank", window=22)

# Save ic chart
rank_chart_path = results_folder / "rank_ic_chart.png"
pearson_chart_path = results_folder / "pearson_ic_chart.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
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