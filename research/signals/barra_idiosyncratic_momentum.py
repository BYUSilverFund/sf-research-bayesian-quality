import polars as pl

def barra_idiosyncratic_momentum() -> pl.Expr:
    return(
        pl.col("specific_return")
        .log1p()
        .rolling_sum(230)
        .shift(21)
        .over("barrid")
        .alias('barra_momentum')
    )