"""観測済みの履歴で起点を選び、未来はラベルと選択条件にだけ使う。"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .data import connection


def build_pairs():
    """T1/T2 と organic 初接触の比較用 O2。全件を pandas に読み込まない。"""
    raw_path = config.RAW / "flat/50m/listens.parquet"
    target = config.PROCESSED / "pairs.parquet"
    with connection() as con:
        con.from_parquet(str(raw_path)).create_view("raw_events")
        con.execute("CREATE VIEW events AS SELECT uid, item_id, timestamp::BIGINT AS timestamp, "
                    "is_organic, played_ratio_pct FROM raw_events")
        max_timestamp = con.execute("SELECT max(timestamp) FROM events").fetchone()[0]
        split_case = "CASE " + " ".join(
            f"WHEN timestamp // {config.SECONDS_PER_DAY} BETWEEN {lo} AND {hi} THEN '{name}'"
            for name, (lo, hi) in config.SPLITS.items()
        ) + " END"
        con.execute("CREATE TABLE firsts AS SELECT uid, item_id, min(timestamp) AS first_timestamp "
                    "FROM events GROUP BY uid,item_id")
        # 同一5秒ビンに経路違いがある初接触は、順序を推測せず除外する。
        con.execute(f"""CREATE TABLE anchors AS
            WITH first_events AS (
                SELECT e.uid, e.item_id, e.timestamp AS anchor_timestamp,
                       min(e.is_organic) AS route, min(e.played_ratio_pct) AS played_ratio_pct,
                       min(e.is_organic) = max(e.is_organic) AS unambiguous
                FROM events e JOIN firsts f USING (uid,item_id)
                WHERE e.timestamp = f.first_timestamp
                GROUP BY e.uid,e.item_id,e.timestamp
            ), first_contacts AS (
                SELECT *, {split_case.replace('timestamp //', 'anchor_timestamp //')} AS split
                FROM first_events WHERE unambiguous
            ), recommendations AS (
                SELECT uid,item_id,timestamp,played_ratio_pct,{split_case} AS split
                FROM events WHERE is_organic=0
            ), first_recommendations AS (
                SELECT uid,item_id,split,min(timestamp) AS anchor_timestamp,
                       arg_min(played_ratio_pct, struct_pack(t := timestamp, r := played_ratio_pct)) AS played_ratio_pct
                FROM recommendations WHERE split IS NOT NULL GROUP BY uid,item_id,split
            )
            SELECT uid,item_id,anchor_timestamp,played_ratio_pct,split,
                   CASE WHEN route=0 THEN 'T2' ELSE 'O2' END AS task
            FROM first_contacts WHERE split IS NOT NULL
            UNION ALL
            SELECT r.uid,r.item_id,r.anchor_timestamp,r.played_ratio_pct,r.split,'T1' AS task
            FROM first_recommendations r JOIN firsts f USING(uid,item_id)
            WHERE f.first_timestamp < r.anchor_timestamp
        """)
        con.execute(f"""COPY (
            SELECT a.*, a.anchor_timestamp // {config.SECONDS_PER_DAY} AS anchor_day,
                   a.anchor_timestamp + {config.HORIZON_DAYS * config.SECONDS_PER_DAY} <= {max_timestamp} AS complete_followup,
                   min(e.timestamp-a.anchor_timestamp) AS first_relisten_seconds_b,
                   min(e.timestamp-a.anchor_timestamp) FILTER (WHERE e.is_organic=1) AS first_relisten_seconds_c,
                   list(DISTINCT e.timestamp-a.anchor_timestamp ORDER BY e.timestamp-a.anchor_timestamp)
                       FILTER (WHERE e.timestamp IS NOT NULL) AS relisten_seconds_b,
                   list(DISTINCT e.timestamp-a.anchor_timestamp ORDER BY e.timestamp-a.anchor_timestamp)
                       FILTER (WHERE e.is_organic=1) AS relisten_seconds_c
            FROM anchors a LEFT JOIN events e ON a.uid=e.uid AND a.item_id=e.item_id
                AND e.timestamp>a.anchor_timestamp
                AND e.timestamp<=a.anchor_timestamp+{config.HORIZON_DAYS * config.SECONDS_PER_DAY}
            GROUP BY ALL
            ORDER BY task,split,uid,item_id,anchor_timestamp
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        excluded = con.execute("""SELECT count(*) FROM (
            SELECT e.uid,e.item_id FROM events e JOIN firsts f USING(uid,item_id)
            WHERE e.timestamp=f.first_timestamp GROUP BY e.uid,e.item_id
            HAVING min(e.is_organic)<>max(e.is_organic))""").fetchone()[0]
    (config.PROCESSED / "pair_provenance.json").write_text(json.dumps({
        "max_timestamp": max_timestamp, "ambiguous_first_contact_pairs_excluded": excluded,
        "split_ranges": config.SPLITS, "window": "elapsed seconds: [L*86400, H*86400]",
        "source_revision": json.loads((config.RAW / "manifest.json").read_text())["revision"],
    }, indent=2) + "\n")
    return target


def activity_counts(pairs, user_days, lower_days, horizon_days):
    """同じ経過時間窓内の、全経路の再生がある暦日数。境界日は時刻で判定。"""
    if not 0 <= lower_days < horizon_days:
        raise ValueError("0 <= L < H が必要")
    users = np.sort(user_days["uid"].unique().to_numpy())
    u = np.searchsorted(users, user_days["uid"].to_numpy())
    pu = np.searchsorted(users, pairs["uid"].to_numpy())
    if np.any(pu >= len(users)) or np.any(users[pu] != pairs["uid"].to_numpy()):
        raise ValueError("活動表に起点ユーザーがない")
    day = user_days["day"].to_numpy().astype(np.int64)
    size = int(max(day.max(), ((pairs["anchor_timestamp"].max() + horizon_days * config.SECONDS_PER_DAY) // config.SECONDS_PER_DAY))) + 2
    first = np.full((len(users), size), np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full((len(users), size), -1, dtype=np.int64)
    first[u, day] = user_days["first_timestamp"].to_numpy()
    last[u, day] = user_days["last_timestamp"].to_numpy()
    active = last >= 0
    prefix = np.pad(np.cumsum(active, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    anchor = pairs["anchor_timestamp"].to_numpy().astype(np.int64)
    start, end = anchor + lower_days * config.SECONDS_PER_DAY, anchor + horizon_days * config.SECONDS_PER_DAY
    sd, ed = start // config.SECONDS_PER_DAY, end // config.SECONDS_PER_DAY
    return (prefix[pu, ed] - prefix[pu, sd + 1]
            + (last[pu, sd] >= start) + (first[pu, ed] <= end)).astype(np.int16)


def label_expr(route, lower_days, horizon_days):
    return pl.col(f"relisten_seconds_{route}").list.eval(
        (pl.element() >= lower_days * config.SECONDS_PER_DAY)
        & (pl.element() <= horizon_days * config.SECONDS_PER_DAY)
    ).list.any().fill_null(False).cast(pl.UInt8).alias(f"label_{route}")


def with_labels(pairs, lower_days=None, horizon_days=config.HORIZON_DAYS, min_active_days=30):
    if lower_days is None:
        lower_days = config.LOWER_BOUND_DAYS
    if not 0 <= lower_days < horizon_days <= config.HORIZON_DAYS:
        raise ValueError("0 <= L < H <= 90 が必要")
    return pairs.filter(
        pl.col(f"complete_followup_{horizon_days}")
        & (pl.col(f"active_days_l{lower_days}_h{horizon_days}") >= min_active_days)
    ).with_columns(label_expr("b", lower_days, horizon_days), label_expr("c", lower_days, horizon_days))


def prepare_labels():
    pair_path = build_pairs()
    pairs = pl.read_parquet(pair_path)
    days = pl.read_parquet(config.PROCESSED / "user_days.parquet")
    provenance = json.loads((config.PROCESSED / "pair_provenance.json").read_text())
    for horizon in (60, config.HORIZON_DAYS):
        pairs = pairs.with_columns(
            (pl.col("anchor_timestamp") + horizon * config.SECONDS_PER_DAY <= provenance["max_timestamp"])
            .alias(f"complete_followup_{horizon}")
        )
        for lower in sorted(set(config.SENSITIVITY_LOWER_DAYS) | {config.LOWER_BOUND_DAYS}):
            pairs = pairs.with_columns(pl.Series(
                f"active_days_l{lower}_h{horizon}", activity_counts(pairs, days, lower, horizon)
            ))
    pairs.write_parquet(pair_path, compression="zstd")
    main = with_labels(pairs).filter(pl.col("task").is_in(["T1", "T2"]))
    main.write_parquet(config.PROCESSED / "main_labels.parquet", compression="zstd")
    return write_summary_tables(pairs)


def summarize(pairs):
    return pairs.group_by("task", "split").agg(
        pl.len().alias("pairs"), pl.col("uid").n_unique().alias("users"),
        pl.col("label_c").mean().alias("positive_rate_c"),
        pl.col("label_b").mean().alias("positive_rate_b"),
    ).sort("task", "split")


def write_summary_tables(pairs):
    main = summarize(with_labels(pairs).filter(pl.col("task").is_in(["T1", "T2"])))
    preliminary = pl.DataFrame({
        "task": ["T2", "T2", "T2", "T1", "T1", "T1"],
        "split": ["train", "validation", "test"] * 2,
        "preliminary_pairs_approx": [740000, 270000, 940000, 410000, 480000, 960000],
        "preliminary_positive_rate_c": [0.085, 0.10, 0.09, 0.186, 0.191, 0.187],
    })
    main = main.join(preliminary, on=["task", "split"]).with_columns(
        (pl.col("positive_rate_c") - pl.col("preliminary_positive_rate_c")).alias("rate_delta")
    )
    main.write_csv(config.RESULTS / "split_summary.csv")
    tables = []
    for horizon in (60, config.HORIZON_DAYS):
        for lower in sorted(set(config.SENSITIVITY_LOWER_DAYS) | {config.LOWER_BOUND_DAYS}):
            for active in (0, 10, 30):
                table = summarize(with_labels(pairs, lower, horizon, active)).with_columns(
                    pl.lit(lower).alias("lower_days"), pl.lit(horizon).alias("horizon_days"),
                    pl.lit(active).alias("min_active_days"),
                )
                tables.append(table)
    pl.concat(tables).write_csv(config.RESULTS / "label_sensitivity.csv")
    pairs.group_by("task", "split").agg(
        pl.len().alias("all_anchors"), (~pl.col("complete_followup_90")).sum().alias("censored_anchors")
    ).sort("task", "split").write_csv(config.RESULTS / "anchor_counts.csv")
    return main
