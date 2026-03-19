import datetime as dt
import os
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
signal_name = "hierarchical_bma_quality"
signal_name_title = "Hierarchical BMA Barra"
IC = 0.05
gamma = 60
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_7")

# Rolling BMA parameters
dynamic_window_months = 60
prior_window_months = 180 # 15-Year moving window for PIP priors
decay = 0.97
ema_alpha = 0.3 # EMA smoothing parameter for the final expected return
checkpoint_dir = "temp/checkpoints_hierarchical_bma"

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

# Load daily factor returns
daily_factor_returns = sfd.load_factors(start=start, end=end).select(
    ["date"] + quality_factors
)

# Compound daily to monthly factor returns
m_fac_rets = (
    daily_factor_returns.sort("date")
    .group_by_dynamic("date", every="1mo")
    .agg(
        [
            (((pl.col(f).truediv(100).add(1)).product().sub(1)) * 100).alias(f)
            for f in quality_factors
        ]
    )
)

# 12-month trailing average on factor returns for pre-smoothing effect
smooth_factors = [f"{f}_smooth" for f in quality_factors]
m_fac_rets_smoothed = m_fac_rets.with_columns(
    [pl.col(f).rolling_mean(12).alias(f"{f}_smooth") for f in quality_factors]
).drop_nulls()

# Target is the next month's raw return for a specific factor, predicted by smoothed features
timing_ready_df = m_fac_rets_smoothed.with_columns(
    [pl.col(f).shift(-1).alias(f"{f}_target") for f in quality_factors]
).drop_nulls()

unique_dates = (
    timing_ready_df.select("date").unique().sort("date").to_series().to_list()
)

all_combos = []
for k in range(1, len(smooth_factors) + 1):
    all_combos.extend(list(combinations(smooth_factors, k)))

rolling_results = []
prev_smoothed_forecasts = {f: 0.0 for f in quality_factors}

# Helper function to perform OLS and calulate BICs
def get_bics(df_X, y, combos, is_weighted=False):
    """Helper function to run OLS on all models and calculate BICs."""
    n = len(df_X)
    bics = []
    params_list = []
    
    if is_weighted:
        w = df_X.get_column("sqrt_w").to_numpy()
        y_w = y * w
    else:
        y_w = y
        
    yy_w = np.vdot(y_w, y_w)
        
    for subset in combos:
        X_cols = ["const"] + list(subset)
        
        if is_weighted:
            X_w = df_X.select([pl.col(c) * pl.col("sqrt_w") for c in X_cols]).to_numpy()
        else:
            X_w = df_X.select(X_cols).to_numpy()
            
        xtx = X_w.T @ X_w
        xty = X_w.T @ y_w
        
        try:
            beta = np.linalg.solve(xtx, xty)
            ssr = max(yy_w - np.vdot(beta, xty), 1e-10)
        except np.linalg.LinAlgError:
            beta, ssr_list, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            ssr = ssr_list[0] if len(ssr_list) > 0 else 1e-10
            
        bic = np.log(n) * len(X_cols) + n * np.log(ssr / n)
        bics.append(bic)
        params_list.append(dict(zip(X_cols, beta)))
        
    return np.array(bics), params_list

# Walk-forward loop for hierarchical factor timing
start_idx = max(prior_window_months, dynamic_window_months)

for i in range(start_idx, len(unique_dates)):
    current_date = unique_dates[i]
    checkpoint_path = f"{checkpoint_dir}/{current_date}.parquet"

    if os.path.exists(checkpoint_path):
        saved_forecasts = pl.read_parquet(checkpoint_path).to_dicts()[0]
        rolling_results.append(saved_forecasts)
        # Keep EMA state updated when loading from checkpoints
        for tf in quality_factors:
            prev_smoothed_forecasts[tf] = saved_forecasts[tf]
        continue

    # Window slicing
    prior_train_dates = unique_dates[i - prior_window_months : i]
    dyn_train_dates = unique_dates[i - dynamic_window_months : i]

    prior_df = timing_ready_df.filter(pl.col("date").is_in(prior_train_dates)).with_columns(pl.lit(1.0).alias("const"))
    dyn_df = timing_ready_df.filter(pl.col("date").is_in(dyn_train_dates))

    weight_map = {
        d: (decay ** (len(dyn_train_dates) - 1 - idx)) for idx, d in enumerate(dyn_train_dates)
    }
    
    dyn_df = dyn_df.with_columns(
        [
            pl.col("date").replace(weight_map).cast(pl.Float64).alias("obs_weights"),
            pl.lit(1.0).alias("const"),
        ]
    ).with_columns(pl.col("obs_weights").sqrt().alias("sqrt_w"))

    month_forecasts = {"date": current_date}
    
    # UPDATED: Pull the features for the actual current date to generate an out-of-sample forecast
    latest_X = timing_ready_df.filter(pl.col("date") == current_date).row(0, named=True)

    for target_factor in quality_factors:
        target_col = f"{target_factor}_target"
        
        # 15-year moving prior
        y_prior = prior_df.get_column(target_col).to_numpy()
        prior_bics, _ = get_bics(prior_df, y_prior, all_combos, is_weighted=False)
        
        prior_bics_adj = prior_bics - np.min(prior_bics)
        prior_pmps = np.exp(-0.5 * prior_bics_adj)
        prior_pmps /= prior_pmps.sum()
        
        # Calculate PIPs from prior window
        pips = {f: 0.0 for f in smooth_factors}
        for idx, subset in enumerate(all_combos):
            for f in subset:
                pips[f] += prior_pmps[idx]
                
        # Convert PIPs to model priors
        model_priors = []
        for subset in all_combos:
            prior_prob = 1.0
            for f in smooth_factors:
                if f in subset:
                    prior_prob *= pips[f]
                else:
                    prior_prob *= (1.0 - pips[f])
            model_priors.append(prior_prob)
            
        model_priors = np.array(model_priors)
        if model_priors.sum() > 0:
            model_priors /= model_priors.sum()
        else:
            model_priors = np.ones(len(all_combos)) / len(all_combos)

        # 5-year rolling likelihood
        y_dyn = dyn_df.get_column(target_col).to_numpy()
        dyn_bics, dyn_params = get_bics(dyn_df, y_dyn, all_combos, is_weighted=True)
        
        dyn_bics_adj = dyn_bics - np.min(dyn_bics)
        dyn_likelihoods = np.exp(-0.5 * dyn_bics_adj)
        
        # Multiply likelihood by informed prior
        hierarchical_pmps = dyn_likelihoods * model_priors
        if hierarchical_pmps.sum() > 0:
            hierarchical_pmps /= hierarchical_pmps.sum()
        else:
            hierarchical_pmps = np.ones(len(all_combos)) / len(all_combos)

        # Calculate expected return
        expected_return = 0.0
        for idx, subset in enumerate(all_combos):
            m_params = dyn_params[idx]
            model_forecast = m_params.get("const", 0.0)
            for f in subset:
                model_forecast += m_params[f] * latest_X[f]

            expected_return += model_forecast * hierarchical_pmps[idx]

        # Output beta smoothing using EMA
        smoothed_return = (ema_alpha * expected_return) + ((1 - ema_alpha) * prev_smoothed_forecasts[target_factor])
        prev_smoothed_forecasts[target_factor] = smoothed_return
        
        month_forecasts[target_factor] = smoothed_return

    pl.DataFrame([month_forecasts]).write_parquet(checkpoint_path)
    rolling_results.append(month_forecasts)

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

# Load factor exposures
factors = sfd.load_exposures(
    start=start, end=end, in_universe=True, columns=["date", "barrid"] + quality_factors
).with_columns(
    [
        (
            (pl.col(f).sub(pl.col(f).mean().over("date"))).truediv(pl.col(f).std().over("date"))
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
        (pl.col(signal_name).sub(pl.col(signal_name).mean().over("date")))
        .truediv(pl.col(signal_name).std().over("date"))
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
        "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("return")
    )
    .drop_nulls("return")
)

# Merge alphas and forward returns
merged = alphas.join(other=forward_returns, on=["date", "barrid"], how="inner")

# Get merged alphas and forward returns (inner join)
merged_alphas = merged.select("date", "barrid", "alpha")
merged_forward_returns = merged.select("date", "barrid", "return")

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