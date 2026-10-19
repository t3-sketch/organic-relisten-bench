import polars as pl

from .config import REPRESENTATION_END_DAY, SPLITS, SECONDS_PER_DAY


def split_expr(day_column="anchor_day"):
    day = pl.col(day_column)
    result = pl.lit(None, dtype=pl.String)
    for name, (start, end) in SPLITS.items():
        result = pl.when(day.is_between(start, end)).then(pl.lit(name)).otherwise(result)
    return result.alias("split")


def assign_splits(pairs):
    return pairs.with_columns(split_expr()).filter(pl.col("split").is_not_null())


def representation_history(listens):
    return listens.filter(pl.col("timestamp").cast(pl.Int64) < (REPRESENTATION_END_DAY + 1) * SECONDS_PER_DAY)
