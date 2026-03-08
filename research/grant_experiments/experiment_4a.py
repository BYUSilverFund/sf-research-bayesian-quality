import datetime as dt
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
is_start = dt.date(1996, 1, 1)
is_end = dt.date(2014, 12, 31)
oos_start = dt.date(2015, 1, 1)
oos_end = dt.date(2024, 12, 31)

price_filter = 5
signal_name = "quality"
signal_name_title = "IS/OOS Static Bayesian Quality"
IC = 0.05
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_4")

# Create folders
results_folder.mkdir(parents=True, exist_ok=True)

# Barra quality factor setup
quality_factors = [
    "USSLOWL_PROFIT",
    "USSLOWL_EARNQLTY",
    "USSLOWL_MGMTQLTY",
    "USSLOWL_LEVERAGE",
    "USSLOWL_GROWTH",
]
request_columns = ["date", "barrid"] + quality_factors

# Load lightweight monthly data for weight calculation
factors_monthly = sfd.load_exposures(
    start=is_start, end=is_end, in_universe=True, columns=request_columns
).with_columns(
    [
        (
            (pl.col(f) - pl.col(f).mean().over("date")) / pl.col(f).std().over("date")
        ).alias(f)
        for f in quality_factors
    ]
)

returns_monthly = sfd.load_assets(
    start=is_start, end=is_end, in_universe=True, columns=["date", "barrid", "return"]
)

df_monthly = returns_monthly.join(factors_monthly, on=["date", "barrid"], how="inner")

monthly_factors = (
    df_monthly.sort("date")
    .group_by_dynamic("date", every="1mo", group_by="barrid")
    .agg([pl.col(f).last() for f in quality_factors])
)

# Aggregating returns
m_rets = (
    df_monthly.sort("date")
    .group_by_dynamic("date", every="1mo", group_by="barrid")
    .agg(((pl.col("return") / 100 + 1).product() - 1).alias("monthly_return"))
).with_columns(pl.col("monthly_return").shift(-1).over("barrid").alias("fwd_return"))

# Join the same month return to the same month exposure
bma_ready_df = monthly_factors.join(
    m_rets.select(["date", "barrid", "fwd_return"]), on=["date", "barrid"], how="inner"
).drop_nulls()

# Run the BMA math once over the entire IS period
window_df = bma_ready_df.with_columns(pl.lit(1.0).alias("const"))
y = window_df.get_column("fwd_return").to_numpy()
yy = np.vdot(y, y)
n = len(window_df)

all_combos = []
for k in range(1, len(quality_factors) + 1):
    all_combos.extend(list(combinations(quality_factors, k)))

model_stats = []
for subset in all_combos:
    subset_list = list(subset)
    X_cols = ["const"] + subset_list
    X = window_df.select(X_cols).to_numpy()

    xtx = X.T @ X
    xty = X.T @ y

    try:
        beta = np.linalg.solve(xtx, xty)
        ssr = max(yy - np.vdot(beta, xty), 1e-10)
    except np.linalg.LinAlgError:
        beta, ssr_list, _, _ = np.linalg.lstsq(X, y, rcond=None)
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

bma_weights = {f: 0.0 for f in quality_factors}
for idx, m in enumerate(model_stats):
    for factor in m["factors"]:
        bma_weights[factor] += m["params"][factor] * pmp[idx]

print("--- IS Weights Calculated ---")
for k, v in bma_weights.items():
    print(f"{k}: {v:.6f}")
print("-----------------------------\n")

# Get daily data for OOS period only
returns = sfd.load_assets(
    start=oos_start,
    end=oos_end,
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
    start=oos_start,
    end=oos_end,
    in_universe=True,
    columns=["date", "barrid"] + quality_factors,
).with_columns(
    [
        (
            (pl.col(f) - pl.col(f).mean().over("date")) / pl.col(f).std().over("date")
        ).alias(f)
        for f in quality_factors
    ]
)

data = returns.join(factors, on=["date", "barrid"], how="inner")

# Compute static signal using OOS daily exposures and IS static weights
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

# Get ics
ics = sfp.generate_alpha_ics(
    alphas=alphas, rets=forward_returns, method="rank", window=22
)

# Save ic charts
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
