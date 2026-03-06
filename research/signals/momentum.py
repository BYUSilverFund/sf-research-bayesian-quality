import polars as pl

def momentum() -> pl.Expr:
    return (
        pl.col("return")
        .log1p()
        .rolling_sum(230)
        .shift(21)
        .over("barrid")
        .alias("momentum")
    )