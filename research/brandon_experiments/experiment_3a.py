# Quality Only - Bayesian Signal Construction

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
train_window = 252 * 2  # 2-year rolling window for the Bayesian model

results_folder.mkdir(parents=True, exist_ok=True)

# 1. Get Data and Exposures
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

merged_data = data.join(exposures, on=["date", "barrid"], how="left")

# 2. Prepare Regression Target (Forward Specific Return)
# We use specific_return to isolate the alpha component.
# Shift by -1 to align today's exposures with tomorrow's return.
prepared_data = (
    merged_data.sort("barrid", "date")
    .with_columns(
        pl.col("specific_return").shift(-1).over("barrid").alias("fwd_specific_return")
    )
    .drop_nulls(
        subset=[
            "USSLOWL_EARNQLTY",
            "USSLOWL_VALUE",
            "USSLOWL_PROFIT",
            "fwd_specific_return",
        ]
    )
)

# 3. Rolling Bayesian Regression
# Polars is great for vectorization, but rolling cross-sectional models
# are often easier to manage by dropping into NumPy/Pandas.
df_pd = prepared_data.select(
    "date",
    "barrid",
    "USSLOWL_EARNQLTY",
    "USSLOWL_VALUE",
    "USSLOWL_PROFIT",
    "fwd_specific_return",
).to_pandas()

unique_dates = np.sort(df_pd["date"].unique())
factors = ["USSLOWL_EARNQLTY", "USSLOWL_VALUE", "USSLOWL_PROFIT"]

# Store daily signals
signal_records = []

# Initialize Bayesian Ridge (Defaults to uninformative priors, adjusts alpha/lambda via empirical Bayes)
model = BayesianRidge(compute_score=True)

print("Running rolling Bayesian regression...")
weights_records = []  # NEW: To track factor balance

# Define your prior beliefs based on your original formula:
# ((EARNQLTY + PROFIT) / 2 + VALUE) / 2
# This equals: 0.25 * EARNQLTY + 0.50 * VALUE + 0.25 * PROFIT
prior_weights = np.array([0.25, 0.50, 0.25])

for i in range(train_window, len(unique_dates)):
    # To avoid lookahead bias, train on data strictly before the current date 't'.
    # Because we aligned 'fwd_specific_return' at t-1 with returns realized at t,
    # training up to t-1 is safe for predicting t+1 using exposures at t.
    train_dates = unique_dates[i - train_window : i]
    current_date = unique_dates[i]

    # Train set
    train_mask = df_pd["date"].isin(train_dates)
    X_train = df_pd.loc[train_mask, factors].values
    y_train = df_pd.loc[train_mask, "fwd_specific_return"].values

    # ---------------------------------------------------------
    # INCORPORATING PRIORS:
    # Subtract the returns explained by your prior weights
    # y_adjusted = y - (X * prior_weights)
    # ---------------------------------------------------------
    expected_y_from_priors = np.dot(X_train, prior_weights)
    y_train_adjusted = y_train - expected_y_from_priors

    # Fit Bayesian Model on the *residuals*
    model.fit(X_train, y_train_adjusted)

    # weights_records.append({
    #     "date": current_date,
    #     "EARNQLTY_weight": model.coef_[0],
    #     "VALUE_weight": model.coef_[1],
    #     "PROFIT_weight": model.coef_[2]
    # })

    # Predict (Generate Signal) for current date
    curr_mask = df_pd["date"] == current_date
    curr_data = df_pd.loc[curr_mask]

    # The true weights are your prior weights PLUS the model's calculated adjustments
    final_weights = model.coef_ + prior_weights

    weights_records.append(
        {
            "date": current_date,
            "EARNQLTY_weight": final_weights[0],
            "VALUE_weight": final_weights[1],
            "PROFIT_weight": final_weights[2],
        }
    )

    if len(curr_data) > 0:
        X_curr = curr_data[factors].values

        # The expected return is our signal value
        # expected_returns = model.predict(X_curr)

        expected_returns = np.dot(X_curr, final_weights)

        # Collect results
        for barrid, signal_val in zip(curr_data["barrid"].values, expected_returns):
            signal_records.append(
                {"date": current_date, "barrid": barrid, signal_name: signal_val}
            )


# Convert weights to a Pandas DataFrame for easy plotting
weights_df = pd.DataFrame(weights_records)

# Generate the visualization
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

# Add a zero line for reference (weights below 0 mean the model is shorting the factor)
plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title("Dynamic Factor Weights over Time (Bayesian Ridge)")
plt.xlabel("Date")
plt.ylabel("Coefficient Weight (Beta)")
plt.legend()
plt.tight_layout()

# Save the chart
weights_chart_path = results_folder / "dynamic_factor_weights.png"
plt.savefig(weights_chart_path)
print(f"Factor weights chart saved to {weights_chart_path}")

# Convert back to Polars
signals_df = pl.DataFrame(signal_records)

# 4. Integrate Signal Back into Main Pipeline
signals = merged_data.join(signals_df, on=["date", "barrid"], how="left")

# Note: Your original code shifted the signal by 2 periods to account for execution delay.
# We apply the same shift here to maintain execution realism.
signals = signals.sort("barrid", "date").with_columns(
    pl.col(signal_name).shift(2).over("barrid").alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col(signal_name).is_not_nan(),
)

# Compute scores (Z-scoring cross-sectionally)
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
ics = sfp.generate_alpha_ics(alphas=alphas, rets=returns, method="rank", window=22)

# Save IC chart
rank_chart_path = results_folder / "rank_ic_chart.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)

# Run parallelized backtest
# run_backtest_parallel(
#     data=alphas,
#     signal_name=signal_name,
#     constraints=constraints,
#     gamma=gamma,
#     n_cpus=n_cpus,
# )
