import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from sklearn.linear_model import BayesianRidge

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
train_window = 252 * 2  # 2-year rolling window (in trading dates)

# train_window = 126  # 2-year rolling window (in trading dates)

# Keep the fast estimator behavior from experiment_3a_fast.py.
# This approximates rolling BayesianRidge by calibrating the shrinkage scale
# on an initial window, then solving rolling ridge systems from sufficient stats.
reestimate_hyperparams_every = (
    None  # e.g. 21 or 63 for periodic refresh; None disables.
)
min_lambda = 1e-10

results_folder.mkdir(parents=True, exist_ok=True)

factor_cols = ["USSLOWL_EARNQLTY", "USSLOWL_VALUE", "USSLOWL_PROFIT"]
prior_weights = np.array([0.25, 0.50, 0.25], dtype=np.float64)


# 1. Load data once.
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
    pl.col("date").cast(pl.Date),
)

exposures = sfd.load_exposures(
    start=start,
    end=end,
    in_universe=True,
    columns=["date", "barrid", *factor_cols],
).with_columns(pl.col("date").cast(pl.Date))

merged_data = data.join(exposures, on=["date", "barrid"], how="left")


# 2. Prepare regression target with explicit date-major ordering.
# Exposures at date t predict specific return realized at date t+1.
prepared_data = (
    merged_data.sort("barrid", "date")
    .with_columns(
        pl.col("specific_return").shift(-1).over("barrid").alias("fwd_specific_return")
    )
    .drop_nulls(subset=[*factor_cols, "fwd_specific_return"])
    .select("date", "barrid", *factor_cols, "fwd_specific_return")
    .with_columns(pl.col("date").cast(pl.Date))
    .sort("date", "barrid")
)

# Extract typed arrays/series column-by-column to avoid object-dtype pollution from
# mixed-type NumPy conversion. This also fixes Polars sort/join failures on Object.
date_dtype = prepared_data.schema["date"]
barrid_dtype = prepared_data.schema["barrid"]

date_all = prepared_data.get_column("date").to_numpy()
barrid_all = prepared_data.get_column("barrid").to_numpy()
X_factors = np.ascontiguousarray(
    prepared_data.select(factor_cols).to_numpy(), dtype=np.float64
)
y_all = np.ascontiguousarray(
    prepared_data.get_column("fwd_specific_return").to_numpy(), dtype=np.float64
)

# Precompute date slices once.
unique_dates, first_idx, counts = np.unique(
    date_all, return_index=True, return_counts=True
)
n_dates = len(unique_dates)

if n_dates <= train_window:
    raise ValueError(
        f"Not enough dates for train_window={train_window}. "
        f"Need more than {train_window}, found {n_dates}."
    )

row_starts = first_idx.astype(np.int64, copy=False)
row_counts = counts.astype(np.int64, copy=False)
row_ends = row_starts + row_counts

# Augment factors with explicit intercept.
p = X_factors.shape[1] + 1
X_aug = np.empty((X_factors.shape[0], p), dtype=np.float64)
X_aug[:, 0] = 1.0
X_aug[:, 1:] = X_factors

# Prior mean in augmented coordinates: [intercept, slopes...].
prior_mean = np.concatenate(([0.0], prior_weights))

# Residualize by the prior-view signal once.
y_adjusted_all = y_all - X_factors @ prior_weights


# 3. Precompute per-date sufficient statistics.
print("Precomputing per-date sufficient statistics...")
Sxx_by_date = np.empty((n_dates, p, p), dtype=np.float64)
Sxy_by_date = np.empty((n_dates, p), dtype=np.float64)

for i in range(n_dates):
    s = row_starts[i]
    e = row_ends[i]
    Xi = X_aug[s:e]
    yi = y_adjusted_all[s:e]
    Sxx_by_date[i] = Xi.T @ Xi
    Sxy_by_date[i] = Xi.T @ yi


# 4. Calibrate shrinkage once from the initial training window.
print("Calibrating Bayesian-style shrinkage...")
init_start = row_starts[0]
init_end = row_starts[train_window]
init_model = BayesianRidge(fit_intercept=False, compute_score=False, copy_X=False)
init_model.fit(X_aug[init_start:init_end], y_adjusted_all[init_start:init_end])
ridge_lambda = max(float(init_model.lambda_) / float(init_model.alpha_), min_lambda)

# Penalize slopes only, not the intercept.
penalty = np.zeros((p, p), dtype=np.float64)
penalty[1:, 1:] = np.eye(p - 1, dtype=np.float64) * ridge_lambda


# 5. Rolling solve with preallocated outputs.
print("Running rolling fast Bayesian-style regression...")
num_weight_rows = n_dates - train_window
num_signal_rows = int(row_counts[train_window:].sum())

signal_dates = np.empty(num_signal_rows, dtype=date_all.dtype)
signal_barrids = np.empty(num_signal_rows, dtype=barrid_all.dtype)
signal_values = np.empty(num_signal_rows, dtype=np.float64)

weight_dates = np.empty(num_weight_rows, dtype=unique_dates.dtype)
weight_values = np.empty((num_weight_rows, p), dtype=np.float64)

Sxx_window = Sxx_by_date[:train_window].sum(axis=0)
Sxy_window = Sxy_by_date[:train_window].sum(axis=0)

signal_ptr = 0
weight_ptr = 0

for i in range(train_window, n_dates):
    if (
        reestimate_hyperparams_every is not None
        and i > train_window
        and (i - train_window) % reestimate_hyperparams_every == 0
    ):
        train_start = row_starts[i - train_window]
        train_end = row_starts[i]
        init_model.fit(
            X_aug[train_start:train_end], y_adjusted_all[train_start:train_end]
        )
        ridge_lambda = max(
            float(init_model.lambda_) / float(init_model.alpha_), min_lambda
        )
        penalty.fill(0.0)
        penalty[1:, 1:] = np.eye(p - 1, dtype=np.float64) * ridge_lambda

    coef_adjusted = np.linalg.solve(Sxx_window + penalty, Sxy_window)
    final_coef = prior_mean + coef_adjusted

    current_date = unique_dates[i]
    weight_dates[weight_ptr] = current_date
    weight_values[weight_ptr] = final_coef
    weight_ptr += 1

    s = row_starts[i]
    e = row_ends[i]
    n_i = e - s

    signal_dates[signal_ptr : signal_ptr + n_i] = date_all[s:e]
    signal_barrids[signal_ptr : signal_ptr + n_i] = barrid_all[s:e]
    signal_values[signal_ptr : signal_ptr + n_i] = X_aug[s:e] @ final_coef
    signal_ptr += n_i

    if i < n_dates - 1:
        out_idx = i - train_window
        in_idx = i
        Sxx_window += Sxx_by_date[in_idx] - Sxx_by_date[out_idx]
        Sxy_window += Sxy_by_date[in_idx] - Sxy_by_date[out_idx]


def _typed_series(name: str, values, dtype: pl.DataType) -> pl.Series:
    # Construct via Python list when needed so Polars sees a native logical dtype
    # instead of an opaque Object array.
    try:
        return pl.Series(name, values, dtype=dtype, strict=False)
    except Exception:
        return pl.Series(name, values.tolist(), dtype=dtype, strict=False)


# 6. Diagnostics / weights chart.
weights_df = pl.DataFrame(
    {
        "date": _typed_series("date", weight_dates, date_dtype),
        "intercept": weight_values[:, 0],
        "EARNQLTY_weight": weight_values[:, 1],
        "VALUE_weight": weight_values[:, 2],
        "PROFIT_weight": weight_values[:, 3],
    }
)

weights_plot = weights_df.sort("date")
plt.figure(figsize=(12, 6))
plt.plot(
    weights_plot["date"].to_list(),
    weights_plot["EARNQLTY_weight"].to_list(),
    label="Earnings Quality",
    alpha=0.8,
)
plt.plot(
    weights_plot["date"].to_list(),
    weights_plot["VALUE_weight"].to_list(),
    label="Value",
    alpha=0.8,
)
plt.plot(
    weights_plot["date"].to_list(),
    weights_plot["PROFIT_weight"].to_list(),
    label="Profitability",
    alpha=0.8,
)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Dynamic Factor Weights over Time (Fast Bayesian-Style Ridge)")
plt.xlabel("Date")
plt.ylabel("Coefficient Weight (Beta)")
plt.legend()
plt.tight_layout()

weights_chart_path = results_folder / "dynamic_factor_weights_faster_structural.png"
plt.savefig(weights_chart_path, dpi=150)
plt.close()
print(f"Factor weights chart saved to {weights_chart_path}")


# 7. Build the compact signal table only once.
signals_df = pl.DataFrame(
    {
        "date": _typed_series("date", signal_dates, date_dtype),
        "barrid": _typed_series("barrid", signal_barrids, barrid_dtype),
        signal_name: signal_values,
    }
)


# 8. Join only the columns needed for downstream construction.
base_panel = merged_data.select(
    "date", "barrid", "price", "return", "specific_risk", "predicted_beta"
).with_columns(pl.col("date").cast(pl.Date))

signals = (
    base_panel.join(signals_df, on=["date", "barrid"], how="left")
    .sort("barrid", "date")
    .with_columns(pl.col(signal_name).shift(2).over("barrid").alias(signal_name))
)


# 9. Filter universe and produce alphas.
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col(signal_name).is_not_nan(),
)

scores = filtered.with_columns(
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean().over("date"))
    .truediv(pl.col(signal_name).std().over("date"))
    .alias("score")
).select("date", "barrid", "predicted_beta", "specific_risk", "score")

alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

returns = data.select("date", "barrid", "return").sort("date", "barrid")


# 10. Diagnostics / IC chart.
ics = sfp.generate_alpha_ics(
    alphas=alphas,
    rets=returns,
    method="rank",
    window=22,
)

rank_chart_path = results_folder / "rank_ic_chart_faster_structural.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)


# 11. Save reproducibility artifacts.
weights_out = results_folder / "dynamic_factor_weights_faster_structural.csv"
weights_df.write_csv(weights_out)
print(f"Weights saved to {weights_out}")

# Optional backtest
# run_backtest_parallel(
#     data=alphas,
#     signal_name=signal_name,
#     constraints=constraints,
#     gamma=gamma,
#     n_cpus=n_cpus,
# )
