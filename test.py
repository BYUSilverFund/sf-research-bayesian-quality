import datetime as dt
import os
from pathlib import Path

import sf_quant.data as sfd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1997, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "bayesian_quality_factor_return"
signal_name_title = "Factor Return Bayesian Barra"
IC = 0.05
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_5")

# Rolling BMA parameters
window_months = 120
decay = 0.97
checkpoint_dir = "temp/checkpoints_bma_factor_120"

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

print(daily_factor_returns)
