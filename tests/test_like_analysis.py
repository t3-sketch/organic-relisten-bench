import numpy as np
import polars as pl
import pytest

from relisten.like_analysis import label_like_windows, like_group_expr, support_gate, auc_comparison
from relisten.like_analysis import rank_analysis, MODELS
from relisten.like_analysis import _leaders
from unittest.mock import patch
from pathlib import Path


D = 86400


@pytest.mark.parametrize("shifted", [False, True])
def test_registration_preserves_record_on_main_without_unrelated_branch_entry(tmp_path, shifted):
    from relisten.like_analysis import verify_registration
    original = (Path(__file__).parents[1] / "results/like_analysis/registered_decisions_snapshot.md").read_bytes()
    lines = original.splitlines(keepends=True)
    active = b"".join(lines[:223] + lines[232:]) if shifted else original
    decisions, snapshot = tmp_path / "DECISIONS.md", tmp_path / "snapshot.md"
    decisions.write_bytes(active)
    snapshot.write_bytes(original)
    assert verify_registration(decisions, snapshot) == {
        "p13_line": 226 if shifted else 235, "p14_line": 238 if shifted else 247}
    decisions.write_bytes(active.replace("seed=20261016".encode(), "seed=20261017".encode(), 1))
    with pytest.raises(ValueError, match="P13/P14"):
        verify_registration(decisions, snapshot)
    decisions.write_bytes(active)
    snapshot.write_bytes(original + b"changed")
    if shifted:
        with pytest.raises(ValueError, match="snapshot"):
            verify_registration(decisions, snapshot)


@pytest.mark.parametrize("offset, prior, within24, within90", [
    (-5, True, False, False), (0, False, True, True), (5, False, True, True),
    (D - 5, False, True, True),
    (D, False, True, True), (D + 5, False, False, True),
    (90 * D - 5, False, False, True), (90 * D, False, False, True), (90 * D + 5, False, False, False),
])
def test_like_closed_boundaries_and_strict_prior(offset, prior, within24, within90):
    pairs = pl.DataFrame({"uid": [1], "item_id": [10], "anchor_timestamp": [100 * D]})
    likes = pl.DataFrame({"uid": [1], "item_id": [10], "timestamp": [100 * D + offset]})
    result = label_like_windows(pairs, likes)
    assert result.select("prior_like", "like_24h", "like_90d").row(0) == (prior, within24, within90)


def test_prior_group_precedence_and_duplicate_route_independence():
    pairs = pl.DataFrame({"uid": [1], "item_id": [10], "anchor_timestamp": [10 * D]})
    likes = pl.DataFrame({"uid": [1, 1, 1, 1], "item_id": [10] * 4,
                          "timestamp": [9 * D, 10 * D, 10 * D, 11 * D], "is_organic": [0, 0, 1, 0]})
    result = label_like_windows(pairs, likes).with_columns(like_group_expr("like_24h"))
    assert result["prior_like"][0] and result["like_24h"][0]
    assert result["like_group"][0] == "prior_like"
    assert label_like_windows(pairs, likes).equals(label_like_windows(pairs, likes.unique(subset=["uid", "item_id", "timestamp"])))


def test_uid_item_matching_empty_likes_and_future_append():
    pairs = pl.DataFrame({"uid": [1, 1, 2], "item_id": [10, 20, 10], "anchor_timestamp": [D] * 3})
    likes = pl.DataFrame({"uid": [1, 2], "item_id": [10, 20], "timestamp": [D, D]})
    result = label_like_windows(pairs, likes)
    assert result["like_24h"].to_list() == [True, False, False]
    future = pl.concat([likes, pl.DataFrame({"uid": [1], "item_id": [20], "timestamp": [92 * D]})])
    assert result.equals(label_like_windows(pairs, future))
    empty = label_like_windows(pairs, likes.head(0))
    assert not empty["prior_like"].any() and not empty["like_90d"].any()


def test_invalid_timestamp_rejected():
    pairs = pl.DataFrame({"uid": [1], "item_id": [10], "anchor_timestamp": [0]})
    likes = pl.DataFrame({"uid": [1], "item_id": [10], "timestamp": [-1]})
    with pytest.raises(ValueError):
        label_like_windows(pairs, likes)


def test_gate_counts_users_after_prior_exclusion_not_pairs():
    frame = pl.DataFrame({"uid": [1, 1, 2], "task": ["T2"] * 3, "split": ["test"] * 3,
                          "prior_like": [False, False, True], "like_90d": [True] * 3})
    status = support_gate(frame)
    assert status["d_positive_users"] == 1 and status["status"] == "needs_approval"


def test_seed_average_auc_and_paired_common_users():
    # uid2 has no D negatives and must be excluded from D and paired D-C, but not from C.
    frame = pl.DataFrame({"pair_id": list(range(6)), "uid": [1, 1, 1, 2, 2, 2],
                          "like_90d": [0, 1, 1, 1, 1, 1], "label_c": [0, 1, 0, 0, 1, 0],
                          "label_b": [0, 1, 1, 0, 1, 1],
                          "s1": [0., 1., 2., 0., 1., 2.], "s2": [2., 1., 0., 2., 1., 0.],
                          "s3": [0., 1., 2., 0., 1., 2.]})
    metrics, diff, tables, users = auc_comparison(frame, ["s1", "s2", "s3"], "like_90d")
    d = next(row for row in metrics if row["route"] == "D")
    assert d["auc_users"] == 1 and d["positive_users"] == 2 and d["reference_only"]
    assert np.isclose(d["user_auc"], 2 / 3)
    assert tables["C"].height == 2 and users["uid"].to_list() == [1]
    assert diff["paired_users"] == 1 and np.isclose(diff["mean_auc_difference"], 1 / 6)
    assert diff["difference_ci_low"] == diff["difference_ci_high"] == diff["mean_auc_difference"]


def test_no_auc_users_is_missing_not_half():
    frame = pl.DataFrame({"pair_id": [0, 1], "uid": [1, 1], "like_90d": [1, 1],
                          "label_c": [1, 1], "label_b": [1, 1], "score": [0., 1.]})
    metrics, diff, _, _ = auc_comparison(frame, ["score"], "like_90d")
    assert all(row["auc_users"] == 0 and row["user_auc"] is None for row in metrics)
    assert diff["paired_users"] == 0 and np.isnan(diff["mean_auc_difference"])


def test_missing_a_does_not_remove_d_c_models_and_reference_checks_both_labels():
    rows = [{"task": "T2", "split": "test", "selection": "all_pairs", "window": "like_90d",
             "model": model, "route": route, "user_auc": .5 + i / 100,
             "auc_users": 400 if route == "C" else 600}
            for i, model in enumerate(MODELS) for route in ("D", "C", "B")]
    a = pl.DataFrame({"model": list(MODELS[:-1]), "seed": [42] * 7,
                      "ndcg_at_10": [.1 + i / 100 for i in range(7)]})
    with patch("relisten.like_analysis.pl.read_csv", return_value=a):
        result = rank_analysis(pl.DataFrame(rows))["p14_kendall"]
    dc = result.filter(pl.col("comparison") == "D_vs_C").row(0, named=True)
    da = result.filter(pl.col("comparison") == "D_vs_A").row(0, named=True)
    assert dc["models"] == 8 and np.isclose(dc["kendall_tau"], 1) and dc["reference_only"]
    assert da["models"] == 7 and da["kendall_tau"] is None


def test_leaders_handles_no_auc_and_ties():
    absent = pl.DataFrame({"model": ["a", "b"], "D": [None, float("nan")]})
    assert _leaders(absent, "D") == "算出不能"
    tied = pl.DataFrame({"model": ["b", "a", "c"], "D": [.7, .7, .6]})
    assert _leaders(tied, "D") == "a / b"
