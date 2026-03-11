import polars as pl

quality_factors = [
    "USSLOWL_PROFIT",
    "USSLOWL_EARNQLTY",
    "USSLOWL_MGMTQLTY",
    "USSLOWL_LEVERAGE",
    "USSLOWL_GROWTH",
]


def rolling_bma() -> pl.Expr:
    return pl.sum_horizontal(
        [pl.col(f) * pl.col(f + "_beta") for f in quality_factors]
    ).alias("rolling_bma")
