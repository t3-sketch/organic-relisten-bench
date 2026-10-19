import json

import numpy as np
import polars as pl
import pytest

from relisten import config
from relisten.labels import activity_counts, build_pairs, label_expr, with_labels
from relisten.splits import assign_splits, representation_history


def test_split_boundaries_and_representation_cutoff():
    days = [29, 30, 59, 60, 149, 150, 164, 165, 210, 211]
    pairs = pl.DataFrame({"anchor_day": days})
    assert assign_splits(pairs).rows() == [(30, "train"), (59, "train"), (150, "validation"),
                                          (164, "validation"), (165, "test"), (210, "test")]
    events = pl.DataFrame({"timestamp": [149 * config.SECONDS_PER_DAY,
                                        150 * config.SECONDS_PER_DAY - 1,
                                        150 * config.SECONDS_PER_DAY]})
    assert representation_history(events).height == 2


def test_lower_upper_boundaries_and_later_return_after_early_return():
    d = config.SECONDS_PER_DAY
    pairs = pl.DataFrame({"relisten_seconds_b": [[d, 8*d], [7*d], [7*d+1], [90*d], [90*d+1], None],
                          "relisten_seconds_c": [[d, 8*d], [7*d], [7*d+1], [90*d], [90*d+1], None]},
                         schema={"relisten_seconds_b": pl.List(pl.Int64), "relisten_seconds_c": pl.List(pl.Int64)})
    assert pairs.select(label_expr("c", 7, 90)).to_series().to_list() == [1, 1, 1, 1, 0, 0]
    assert pairs.select(label_expr("b", 1, 60)).to_series().to_list() == [1, 1, 1, 0, 0, 0]


def test_activity_window_boundary_days():
    d = config.SECONDS_PER_DAY
    anchor = 30 * d + d//2
    pairs = pl.DataFrame({"uid": [1, 2], "anchor_timestamp": [anchor, anchor]})
    # 同じ暦日の窓外イベントを利用日として数えてはいけない。
    events = pl.DataFrame({"uid": [1]*7+[2], "timestamp": [anchor+7*d, anchor+7*d+1,
        anchor+8*d, anchor+90*d, anchor+90*d+1, anchor+d, anchor+91*d, anchor+d]})
    days = events.with_columns((pl.col("timestamp")//d).alias("day")).group_by("uid", "day").agg(
        pl.col("timestamp").min().alias("first_timestamp"),
        pl.col("timestamp").max().alias("last_timestamp"))
    assert activity_counts(pairs, days, 7, 90).tolist() == [3, 0]
    assert activity_counts(pairs, days, 1, 60).tolist() == [3, 1]


@pytest.fixture
def toy_pairs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    (raw / "flat/50m").mkdir(parents=True)
    processed.mkdir()
    monkeypatch.setattr(config, "RAW", raw)
    monkeypatch.setattr(config, "PROCESSED", processed)
    # connection の保存先はテストの一時領域に限定する。
    import relisten.data as data
    monkeypatch.setattr(data, "PROCESSED", processed)
    (raw / "manifest.json").write_text(json.dumps({"revision": "test"}))
    d = config.SECONDS_PER_DAY
    rows = [(1, 10, 30*d, 0), (1, 10, 31*d, 0), (1, 10, 38*d, 1),
            (1, 11, 10*d, 1), (1, 11, 32*d, 0), (1, 11, 33*d, 0),
            (1, 12, 40*d, 1), (1, 12, 45*d, 0), (1, 12, 50*d, 1),
            (2, 20, 150*d, 0), (2, 20, 158*d, 0), (2, 20, 159*d, 1),
            (2, 20, 165*d, 0), (2, 20, 173*d, 1),
            (2, 21, 210*d, 0), (2, 21, 300*d, 1),
            (3, 30, 40*d, 0), (3, 30, 40*d, 1)]
    frame = pl.DataFrame(rows, schema=["uid", "item_id", "timestamp", "is_organic"], orient="row").with_columns(
        pl.lit(100).alias("played_ratio_pct"))
    frame.write_parquet(raw / "flat/50m/listens.parquet")
    return pl.read_parquet(build_pairs())


def test_t2_t1_and_organic_first_anchors(toy_pairs):
    anchors = set(toy_pairs.select("uid", "item_id", "anchor_day", "task", "split").rows())
    assert (1, 10, 30, "T2", "train") in anchors
    assert (1, 10, 31, "T1", "train") not in anchors
    assert (1, 11, 32, "T1", "train") in anchors
    assert (1, 11, 33, "T1", "train") not in anchors
    assert (1, 12, 40, "O2", "train") in anchors
    assert (1, 12, 45, "T1", "train") in anchors
    assert (2, 20, 165, "T1", "test") in anchors
    assert not any(r[0] == 3 and r[3] == "T2" for r in anchors)


def test_routes_and_exact_followup(toy_pairs):
    d = config.SECONDS_PER_DAY
    pair = toy_pairs.filter((pl.col("item_id") == 20) & (pl.col("task") == "T2")).row(0, named=True)
    assert pair["first_relisten_seconds_b"] == 8*d
    assert pair["first_relisten_seconds_c"] == 9*d
    assert pair["complete_followup"] is True
    assert toy_pairs.filter(pl.col("item_id") == 21)["first_relisten_seconds_c"][0] == 90*d


def test_activity_filter_and_censoring():
    d = config.SECONDS_PER_DAY
    pairs = pl.DataFrame({"relisten_seconds_b": [[8*d]]*4, "relisten_seconds_c": [[8*d]]*4,
        "complete_followup_90": [True, True, True, False],
        f"active_days_l{config.LOWER_BOUND_DAYS}_h90": [0, 10, 30, 50]})
    assert with_labels(pairs).height == 1
    assert with_labels(pairs, min_active_days=10).height == 2
    assert with_labels(pairs, min_active_days=0).height == 3


def test_vector_activity_matches_brute_force():
    rng = np.random.default_rng(12)
    d = config.SECONDS_PER_DAY
    events = pl.DataFrame({"uid": rng.integers(1, 5, 3000), "timestamp": rng.integers(0, 301*d, 3000)})
    days = events.with_columns((pl.col("timestamp")//d).alias("day")).group_by("uid", "day").agg(
        pl.col("timestamp").min().alias("first_timestamp"), pl.col("timestamp").max().alias("last_timestamp"))
    pairs = pl.DataFrame({"uid": rng.integers(1, 5, 50), "anchor_timestamp": rng.integers(30*d, 210*d, 50)})
    for lower, upper in [(1, 60), (7, 90), (14, 90)]:
        actual = activity_counts(pairs, days, lower, upper)
        expected = []
        for uid, timestamp in pairs.iter_rows():
            eligible = events.filter((pl.col("uid")==uid) & (pl.col("timestamp")>=timestamp+lower*d)
                                     & (pl.col("timestamp")<=timestamp+upper*d))
            expected.append(eligible.select((pl.col("timestamp")//d).n_unique()).item())
        assert actual.tolist() == expected


def test_24_hours_inclusive_and_previous_bin_excluded():
    d = config.SECONDS_PER_DAY
    frame = pl.DataFrame({"relisten_seconds_c": [[d-5], [d], [d+5], [90*d], [90*d+5]]})
    assert frame.select(label_expr("c", 1, 90)).to_series().to_list() == [0, 1, 1, 1, 0]
    pairs = pl.DataFrame({"uid": [1], "anchor_timestamp": [30*d]})
    days = pl.DataFrame({"uid": [1], "day": [31], "first_timestamp": [31*d], "last_timestamp": [31*d]})
    assert activity_counts(pairs, days, 1, 90).tolist() == [1]
