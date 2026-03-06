import polars as pl

def volatility_scaled_momentum() -> pl.Expr:
    return (
        pl.col("return")
        .log1p()
        .rolling_sum(230)
        .truediv(pl.col("return").rolling_std(230))
        .shift(21)
        .over("barrid")
        .alias("vol_scaled_momentum")
    )