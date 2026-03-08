import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from sklearn.linear_model import BayesianRidge

from research.utils import run_backtest_parallel

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "bayesian_qarp"
IC = 0.05
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/brandon/experiment_3")
train_window = 252 * 2  # 2-year rolling window for the Bayesian model

results_folder.mkdir(parents=True, exist_ok=True)

factor_cols = ["USSLOWL_EARNQLTY", "USSLOWL_VALUE", "USSLOWL_PROFIT"]
prior_weights = np.array([0.25, 0.50, 0.25], dtype=np.float64)

# 1. Get Data and Exposures
print("Loading assets and exposures...")
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
    columns=["date", "barrid", *factor_cols],
)

merged_data = data.join(exposures, on=["date", "barrid"], how="left")

# 2. Prepare regression target (forward specific return)
prepared_data = (
    merged_data.sort("barrid", "date")
    .with_columns(
        pl.col("specific_return").shift(-1).over("barrid").alias("fwd_specific_return")
    )
    .drop_nulls(subset=[*factor_cols, "fwd_specific_return"])
    .select("date", "barrid", *factor_cols, "fwd_specific_return")
    .sort("date", "barrid")
)

# Convert once to pandas/numpy with date-major ordering so each rolling window is
# one contiguous row slice instead of a fresh boolean mask over the full table.
df = prepared_data.to_pandas()

# Ensure dense numeric arrays for fast slicing / BLAS ops.
X_all = df[factor_cols].to_numpy(dtype=np.float64, copy=True)
y_all = df["fwd_specific_return"].to_numpy(dtype=np.float64, copy=True)
barrid_all = df["barrid"].to_numpy(copy=True)
date_all = df["date"].to_numpy(copy=True)

# Identify contiguous row blocks for each date.
unique_dates, first_idx, counts = np.unique(
    date_all,
    return_index=True,
    return_counts=True,
)

n_dates = len(unique_dates)
if n_dates <= train_window:
    raise ValueError(
        f"Not enough dates for train_window={train_window}. "
        f"Need more than {train_window}, found {n_dates}."
    )

# Initialize Bayesian Ridge.
# compute_score=False is materially faster and is not needed here.
model = BayesianRidge(
    fit_intercept=True,
    compute_score=False,
    copy_X=False,
)

print("Running rolling Bayesian regression...")

signal_frames = []
weights_records = []

for i in range(train_window, n_dates):
    current_date = unique_dates[i]

    # Training rows correspond to the contiguous block of dates
    # [i-train_window, ..., i-1].
    train_start = first_idx[i - train_window]
    train_end = first_idx[i]  # exclusive

    X_train = X_all[train_start:train_end]
    y_train = y_all[train_start:train_end]

    # Incorporate prior view by fitting only residual alpha unexplained by prior weights.
    y_train_adjusted = y_train - X_train @ prior_weights
    model.fit(X_train, y_train_adjusted)

    final_weights = prior_weights + model.coef_
    final_intercept = model.intercept_

    weights_records.append(
        {
            "date": current_date,
            "EARNQLTY_weight": final_weights[0],
            "VALUE_weight": final_weights[1],
            "PROFIT_weight": final_weights[2],
            "intercept": final_intercept,
        }
    )

    curr_start = first_idx[i]
    curr_end = curr_start + counts[i]
    X_curr = X_all[curr_start:curr_end]

    # Correct prediction path: include intercept from the fitted BayesianRidge model.
    expected_returns = final_intercept + X_curr @ final_weights

    signal_frames.append(
        pd.DataFrame(
            {
                "date": date_all[curr_start:curr_end],
                "barrid": barrid_all[curr_start:curr_end],
                signal_name: expected_returns,
            }
        )
    )

weights_df = pd.DataFrame(weights_records)

# Plot dynamic factor weights
plt.figure(figsize=(12, 6))
plt.plot(
    weights_df["date"],
    weights_df["EARNQLTY_weight"],
    label="Earnings Quality",
    alpha=0.8,
)
plt.plot(weights_df["date"], weights_df["VALUE_weight"], label="Value", alpha=0.8)
plt.plot(
    weights_df["date"], weights_df["PROFIT_weight"], label="Profitability", alpha=0.8
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Dynamic Factor Weights over Time (Bayesian Ridge)")
plt.xlabel("Date")
plt.ylabel("Coefficient Weight (Beta)")
plt.legend()
plt.tight_layout()

weights_chart_path = results_folder / "dynamic_factor_weights.png"
plt.savefig(weights_chart_path, dpi=150)
plt.close()
print(f"Factor weights chart saved to {weights_chart_path}")

if not signal_frames:
    raise ValueError(
        "No signals were generated. Check train_window and available data range."
    )

signals_df = pl.from_pandas(pd.concat(signal_frames, ignore_index=True))

# 4. Integrate signal back into main pipeline
signals = merged_data.join(signals_df, on=["date", "barrid"], how="left")

# Maintain original 2-day execution delay assumption.
signals = signals.sort("barrid", "date").with_columns(
    pl.col(signal_name).shift(2).over("barrid").alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col(signal_name).is_not_nan(),
)

# Compute scores (cross-sectional z-score by date)
scores = filtered.with_columns(
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean().over("date"))
    .truediv(pl.col(signal_name).std().over("date"))
    .alias("score")
).select("date", "barrid", "predicted_beta", "specific_risk", "score")

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

returns = data.sort("date", "barrid").select("date", "barrid", "return")

# Get ICs
ics = sfp.generate_alpha_ics(
    alphas=alphas,
    rets=returns,
    method="rank",
    window=22,
)

rank_chart_path = results_folder / "rank_ic_chart.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)

# Optional: save intermediate outputs for inspection / reproducibility.
weights_out = results_folder / "dynamic_factor_weights.csv"
weights_df.to_csv(weights_out, index=False)
print(f"Weights saved to {weights_out}")

# Optional backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
