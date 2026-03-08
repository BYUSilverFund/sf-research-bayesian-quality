import datetime as dt
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
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
signal_name = "bayesian_barra_rolling_2"
signal_name_title = "Rolling Bayesian Barra"
IC = 0.05
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_3")

# Rolling BMA parameters
window_months = 60
decay = 0.97
checkpoint_dir = "temp/checkpoints_bma_60m"

# Create folders
results_folder.mkdir(parents=True, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Barra quality factor setup
quality_factors = [
    "USSLOWL_PROFIT",
    "USSLOWL_EARNQLTY",
    "USSLOWL_MGMTQLTY",
    "USSLOWL_LEVERAGE",
    "USSLOWL_GROWTH",
]

# Generate rolling weights
request_columns = ["date", "barrid"] + quality_factors

# Load lightweight monthly data for weight calculation
factors_monthly = sfd.load_exposures(
    start=start, end=end, in_universe=True, columns=request_columns
).with_columns(
    [
        (
            (pl.col(f) - pl.col(f).mean().over("date")) / pl.col(f).std().over("date")
        ).alias(f)
        for f in quality_factors
    ]
)

returns_monthly = sfd.load_assets(
    start=start, end=end, in_universe=True, columns=["date", "barrid", "return"]
)

df_monthly = returns_monthly.join(factors_monthly, on=["date", "barrid"], how="inner")

monthly_factors = (
    df_monthly.sort("date")
    .group_by_dynamic("date", every="1mo", group_by="barrid")
    .agg([pl.col(f).last() for f in quality_factors])
)

m_rets = (
    df_monthly.sort("date")
    .group_by_dynamic("date", every="1mo", group_by="barrid")
    .agg(((pl.col("return") / 100 + 1).product() - 1).alias("monthly_return"))
).with_columns(pl.col("monthly_return").shift(-1).over("barrid").alias("fwd_return"))

bma_ready_df = monthly_factors.join(
    m_rets.select(["date", "barrid", "fwd_return"]), on=["date", "barrid"], how="inner"
).drop_nulls()

unique_dates = bma_ready_df.select("date").unique().sort("date").to_series().to_list()

all_combos = []
for k in range(1, len(quality_factors) + 1):
    all_combos.extend(list(combinations(quality_factors, k)))

rolling_results = []

# Walk-forward loop
for i in range(window_months, len(unique_dates)):
    current_date = unique_dates[i]
    checkpoint_path = f"{checkpoint_dir}/{current_date}.parquet"

    if os.path.exists(checkpoint_path):
        rolling_results.append(pl.read_parquet(checkpoint_path).to_dicts()[0])
        continue

    train_dates = unique_dates[i - window_months : i]
    window_df = bma_ready_df.filter(pl.col("date").is_in(train_dates))

    weight_map = {
        d: (decay ** (len(train_dates) - 1 - idx)) for idx, d in enumerate(train_dates)
    }
    window_df = window_df.with_columns(
        [
            pl.col("date").replace(weight_map).cast(pl.Float64).alias("obs_weights"),
            pl.lit(1.0).alias("const"),
        ]
    ).with_columns(pl.col("obs_weights").sqrt().alias("sqrt_w"))

    y_w = (
        window_df.get_column("fwd_return") * window_df.get_column("sqrt_w")
    ).to_numpy()
    yy_w = np.vdot(y_w, y_w)
    n = len(window_df)

    model_stats = []

    for subset in all_combos:
        subset_list = list(subset)
        X_cols = ["const"] + subset_list
        X_w = window_df.select(
            [pl.col(c) * pl.col("sqrt_w") for c in X_cols]
        ).to_numpy()

        xtx = X_w.T @ X_w
        xty = X_w.T @ y_w

        try:
            beta = np.linalg.solve(xtx, xty)
            ssr = max(yy_w - np.vdot(beta, xty), 1e-10)
        except np.linalg.LinAlgError:
            beta, ssr_list, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            ssr = ssr_list[0] if len(ssr_list) > 0 else 1e-10

        model_stats.append(
            {
                "factors": subset_list,
                "bic": np.log(n) * len(X_cols) + n * np.log(ssr / n),
                "params": dict(zip(X_cols, beta)),
            }
        )

    bics = np.array([m["bic"] for m in model_stats])
    bics_adj = bics - np.min(bics)
    pmp = np.exp(-0.5 * bics_adj)
    pmp /= pmp.sum()

    month_betas = {f: 0.0 for f in quality_factors}
    for idx, m in enumerate(model_stats):
        for factor in m["factors"]:
            month_betas[factor] += m["params"][factor] * pmp[idx]

    month_betas["date"] = current_date
    pl.DataFrame([month_betas]).write_parquet(checkpoint_path)
    rolling_results.append(month_betas)

    if i % 12 == 0:
        time.sleep(2)

rolling_weights_df = pl.DataFrame(rolling_results).rename(
    {f: f + "_beta" for f in quality_factors}
)

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
data = data.with_columns(pl.col("date").dt.month_start().alias("month_key"))
weights_with_key = rolling_weights_df.with_columns(
    pl.col("date").dt.offset_by("1mo").dt.month_start().alias("month_key")
).drop("date")

data = data.join(weights_with_key, on="month_key", how="inner")

signals = data.with_columns(
    pl.sum_horizontal([pl.col(f) * pl.col(f + "_beta") for f in quality_factors]).alias(
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
    (
        (pl.col(signal_name) - pl.col(signal_name).mean().over("date"))
        / pl.col(signal_name).std().over("date")
    ).alias("score"),
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
