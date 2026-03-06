import polars as pl

def ff3_idiosyncratic_momentum() -> pl.Expr:
    return (
        pl.col("return_rf")
        .sub(
            pl.sum_horizontal(
                pl.col("B_const"),
                pl.col("B_mkt_rf").mul("mkt_rf"),
                pl.col("B_smb").mul("smb"),
                pl.col("B_hml").mul("hml")
            )
        )
        .log1p()
        .rolling_sum(230)
        .shift(21)
        .over("barrid")
        .alias("ff3_momentum")
    )