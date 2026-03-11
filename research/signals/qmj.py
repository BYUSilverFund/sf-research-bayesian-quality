import polars as pl


def qmj() -> pl.Expr:
    return (
        pl.col("USSLOWL_EARNQLTY")
        .add(pl.col("USSLOWL_PROFIT"))
        .truediv(2)
        .shift(2)
        .over("barrid")
        .alias("qmj")
    )
