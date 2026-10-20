"""P13/P14: 実行前に固定した事後like分析。既存モデル・結果を変更しない。"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kendalltau

from . import config
from .labels import with_labels
from .metrics import bootstrap_mean, paired_user_auc_difference, user_auc_table
from .score_protocol import SCORE_SPECS
from .supplementary import average_seed_auc


SEED = 20261016
BOOTSTRAPS = 1000
WINDOWS = {"like_90d": 90 * 86400, "like_24h": 86400}
MODELS = ("MostPop", "DecayPop", "ItemKNN", "iALS", "BPR", "SASRec", "ACT-R", "AudioCosine")
ROOT = config.ROOT
OUTPUT = config.RESULTS / "like_analysis"
REGISTRATION = {
    "recorded_at_utc": "2026-10-20T06:05:59.767933+00:00",
    "p13_line": 235,
    "p14_line": 247,
    "end_line": 258,
    "decisions_registered_sha256": "42ea83fd9e90e9282615112097109edda25803a4363cd481a8b268097f5b466c",
    "original_prefix_sha256": "e92aaf2c3f8b9f21d6f8114c29fb3de9a06f0a1b30119595229050517a01a111",
    "original_prefix_bytes": 50248,
    "additional_like_analysis_auc_or_label_rates_computed_before_registration": False,
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_registration(decisions, snapshot):
    """行番号がブランチで変わっても、記録時のハッシュと現行のP13/P14を照合する。"""
    active = Path(decisions).read_bytes()
    registered = b"".join(active.splitlines(keepends=True)[:REGISTRATION["end_line"]])
    if hashlib.sha256(registered).hexdigest() != REGISTRATION["decisions_registered_sha256"]:
        registered = Path(snapshot).read_bytes()
        if hashlib.sha256(registered).hexdigest() != REGISTRATION["decisions_registered_sha256"]:
            raise ValueError("実行前記録snapshotが変更された")
    if hashlib.sha256(registered[:REGISTRATION["original_prefix_bytes"]]).hexdigest() != REGISTRATION["original_prefix_sha256"]:
        raise ValueError("実行前記録の既存部分が変更された")
    expected = registered.splitlines(keepends=True)[REGISTRATION["p13_line"] - 1:]
    active_lines = active.splitlines(keepends=True)
    starts = [i for i, line in enumerate(active_lines) if line == expected[0]]
    if len(starts) != 1 or active_lines[starts[0]:starts[0] + len(expected)] != expected:
        raise ValueError("現行DECISIONS.mdのP13/P14定義が実行前記録と異なる")
    return {"p13_line": starts[0] + 1,
            "p14_line": starts[0] + 1 + REGISTRATION["p14_line"] - REGISTRATION["p13_line"]}


def audit_likes():
    """ラベルに結合せず、生イベントの型・単位・重複だけを確認する。"""
    path = config.RAW / "flat/50m/likes.parquet"
    if not path.is_file():
        raise FileNotFoundError("手元の50M likesが必要。自動ダウンロードはしない")
    likes = pl.scan_parquet(path)
    schema = likes.collect_schema()
    required = {"uid", "item_id", "timestamp", "is_organic"}
    if not required.issubset(schema.names()):
        raise ValueError("likesの必要列が不足")
    summary = likes.select(
        pl.len().alias("rows"), pl.col("uid").n_unique().alias("users"),
        pl.col("item_id").n_unique().alias("items"),
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
        ((pl.col("timestamp") % 5) == 0).all().alias("all_multiples_of_5"),
        pl.struct(pl.all()).n_unique().alias("unique_full_rows"),
        pl.struct("uid", "item_id", "timestamp").n_unique().alias("unique_uid_item_timestamp"),
    ).collect().row(0, named=True)
    nulls = likes.select(pl.all().null_count()).collect().row(0, named=True)
    if any(nulls.values()) or summary["min_timestamp"] < 0:
        raise ValueError("likesに欠損または負のtimestampがある")
    repeated = (likes.group_by("uid", "item_id").len().filter(pl.col("len") > 1)
                .select(pl.len().alias("repeated_user_item_pairs"),
                        pl.col("len").sum().alias("rows_in_repeated_pairs"),
                        (pl.col("len") - 1).sum().alias("extra_user_item_events"))
                .collect().row(0, named=True))
    unlike_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").rglob("*")
                          if p.is_file() and "unlike" in p.name.lower())
    if unlike_files:
        raise ValueError("unlikesが新たに存在する。既存の実行前定義を確認し、無断で定義を変えない")
    return {"path": str(path.relative_to(ROOT)), "schema": {k: str(v) for k, v in schema.items()},
            **summary, **repeated, "null_counts": nulls,
            "duplicate_full_rows": summary["rows"] - summary["unique_full_rows"],
            "duplicate_uid_item_timestamp_rows": summary["rows"] - summary["unique_uid_item_timestamp"],
            "timestamp_unit": "seconds; multiples of 5; no multiplication by 5",
            "timestamp_basis": "same Yambda flat revision and seconds convention as listens; DECISIONS.md:7-10,235-245",
            "unlike_files": unlike_files, "unlike_limitation": "取り消しは考慮できない",
            "sha256": file_hash(path)}


def label_like_windows(pairs, likes):
    """uid/item/timestampの辞書順searchで閉区間の存在とstrict priorを返す。"""
    for frame, time_column in ((pairs, "anchor_timestamp"), (likes, "timestamp")):
        for column in ("uid", "item_id", time_column):
            values = frame[column]
            if not values.dtype.is_integer() or values.null_count() or (values < 0).any():
                raise ValueError(f"非負・非欠損の整数が必要: {column}")
    if pairs.height and pairs["anchor_timestamp"].max() > np.iinfo(np.int64).max - WINDOWS["like_90d"]:
        raise ValueError("時間窓の加算がint64を超える")
    dtype = np.dtype([("uid", "u8"), ("item_id", "u8"), ("timestamp", "i8")])
    events = np.empty(likes.height, dtype=dtype)
    events["uid"] = likes["uid"].to_numpy()
    events["item_id"] = likes["item_id"].to_numpy()
    events["timestamp"] = likes["timestamp"].to_numpy()
    events.sort(order=["uid", "item_id", "timestamp"])
    query = np.empty(pairs.height, dtype=dtype)
    query["uid"] = pairs["uid"].to_numpy()
    query["item_id"] = pairs["item_id"].to_numpy()
    anchor = pairs["anchor_timestamp"].to_numpy().astype(np.int64)
    query["timestamp"] = -1
    beginning = np.searchsorted(events, query, side="left")
    query["timestamp"] = anchor
    start = np.searchsorted(events, query, side="left")
    flags = {"prior_like": start > beginning}
    for name, seconds in WINDOWS.items():
        query["timestamp"] = anchor + seconds
        flags[name] = np.searchsorted(events, query, side="right") > start
    result = pairs.with_columns([pl.Series(name, values) for name, values in flags.items()])
    assert result.select((~pl.col("like_24h") | pl.col("like_90d")).all()).item()
    return result


def like_group_expr(window):
    return (pl.when(pl.col("prior_like")).then(pl.lit("prior_like"))
            .when(pl.col(window)).then(pl.lit("like"))
            .otherwise(pl.lit("no_like")).alias("like_group"))


def load_cohort():
    """既存C/Bとpair_idを使い、不要な再聴リストは収集しない。"""
    if config.LOWER_BOUND_DAYS != 1 or config.SECONDS_PER_DAY != 86400:
        raise ValueError("P13/P14はL=1・秒の固定プロトコル")
    pairs = (pl.scan_parquet(config.PROCESSED / "pairs.parquet")
             .filter(pl.col("task").is_in(["T1", "T2"])).with_row_index("pair_id"))
    pairs = with_labels(pairs, lower_days=1, horizon_days=90, min_active_days=30).select(
        "pair_id", "uid", "item_id", "anchor_timestamp", "task", "split", "label_c", "label_b")
    flags = pl.scan_parquet(config.PROCESSED / "seen_item_diagnostics/pair_flags.parquet").select(
        "pair_id", "model_end_day", "seen_in_training")
    frame = pairs.join(flags, on="pair_id", how="left", validate="1:1").collect()
    if frame["pair_id"].n_unique() != frame.height or frame["seen_in_training"].null_count():
        raise ValueError("既存pair IDと既知曲フラグの対応が不正")
    expected = pl.when(pl.col("split") == "train").then(29).otherwise(149)
    assert frame.select((pl.col("model_end_day") == expected).all()).item()
    likes = pl.read_parquet(config.RAW / "flat/50m/likes.parquet", columns=["uid", "item_id", "timestamp"])
    return label_like_windows(frame, likes)


def support_gate(frame):
    """P13の率/P14 AUCより前に、主D90の正例ユーザー数だけを確認する。"""
    primary = frame.filter((pl.col("task") == "T2") & (pl.col("split") == "test") & ~pl.col("prior_like"))
    positive_users = primary.filter(pl.col("like_90d"))["uid"].n_unique()
    return {"task": "T2", "split": "test", "selection": "all_pairs", "window": "like_90d",
            "d_positive_users": positive_users, "required_users": 500,
            "status": "ready" if positive_users >= 500 else "needs_approval"}


def descriptive_analysis(frame):
    strata, reverse, comparisons, user_details = [], [], [], []
    for task in ("T1", "T2"):
        for split in ("train", "validation", "test"):
            sub = frame.filter((pl.col("task") == task) & (pl.col("split") == split))
            for window in WINDOWS:
                keyed = sub.with_columns(like_group_expr(window))
                key = {"analysis": "P13", "task": task, "split": split, "window": window}
                for group in ("prior_like", "like", "no_like"):
                    part = keyed.filter(pl.col("like_group") == group)
                    c, b = int(part["label_c"].sum()), int(part["label_b"].sum())
                    strata.append({**key, "like_group": group, "pairs": len(part), "users": part["uid"].n_unique(),
                                   "c_positives": c, "b_positives": b,
                                   "c_positive_rate": c / len(part) if len(part) else None,
                                   "b_positive_rate": b / len(part) if len(part) else None})
                positives = keyed.filter(pl.col("label_c") == 1)
                nonprior = positives.filter(~pl.col("prior_like"))
                raw_liked = positives.filter(pl.col(window)).height
                new_liked = nonprior.filter(pl.col(window)).height
                reverse.append({**key, "c_positive_pairs_all": positives.height,
                                "c_positive_pairs_prior_like": positives.filter(pl.col("prior_like")).height,
                                "c_positive_pairs_nonprior": nonprior.height,
                                "c_positive_with_window_like_all": raw_liked,
                                "c_positive_with_new_like": new_liked,
                                "window_like_fraction_among_all_c_positive": raw_liked / len(positives) if len(positives) else None,
                                "new_like_fraction_among_all_c_positive": new_liked / len(positives) if len(positives) else None,
                                "new_like_fraction_among_nonprior_c_positive": new_liked / len(nonprior) if len(nonprior) else None})
                new = keyed.filter(~pl.col("prior_like"))
                rates = new.group_by("uid", window).agg(pl.len().alias("n"), pl.col("label_c").mean().alias("rate"))
                yes = rates.filter(pl.col(window)).select("uid", pl.col("n").alias("like_pairs"), pl.col("rate").alias("like_c_rate"))
                no = rates.filter(~pl.col(window)).select("uid", pl.col("n").alias("no_like_pairs"), pl.col("rate").alias("no_like_c_rate"))
                paired = yes.join(no, on="uid", validate="1:1").sort("uid").with_columns(
                    (pl.col("like_c_rate") - pl.col("no_like_c_rate")).alias("difference"))
                low, high = bootstrap_mean(paired["difference"], BOOTSTRAPS, SEED)
                comparisons.append({**key, "paired_users": len(paired),
                                    "mean_like_c_rate": paired["like_c_rate"].mean(),
                                    "mean_no_like_c_rate": paired["no_like_c_rate"].mean(),
                                    "mean_difference": paired["difference"].mean(),
                                    "ci_low": low, "ci_high": high,
                                    "interpretation": "関連のみ。因果効果・好みの裏付けとは解釈しない"})
                user_details.append(paired.with_columns([pl.lit(v).alias(k) for k, v in key.items()]))
    return {"p13_strata": pl.DataFrame(strata), "p13_reverse": pl.DataFrame(reverse),
            "p13_within_user": pl.DataFrame(comparisons), "p13_user_differences": pl.concat(user_details)}


def auc_comparison(frame, score_columns, window):
    """同じペア上のD/C/B。シードをユーザー内で平均してから対応差を取る。"""
    if frame["pair_id"].n_unique() != len(frame):
        raise ValueError("AUCのペアIDが重複")
    if any(not np.isfinite(frame[c].to_numpy()).all() for c in score_columns):
        raise ValueError("保存生スコアに非有限値")
    labels = {"D": window, "C": "label_c", "B": "label_b"}
    tables = {}
    metrics = []
    for route, column in labels.items():
        seeds = [user_auc_table(frame["uid"], frame[column], frame[c].to_numpy()) for c in score_columns]
        table = average_seed_auc(seeds)
        tables[route] = table
        low, high = bootstrap_mean(table["auc"], BOOTSTRAPS, SEED)
        metrics.append({"route": route, "pairs": len(frame), "users": frame["uid"].n_unique(),
                        "positive_pairs": int(frame[column].sum()),
                        "positive_users": frame.filter(pl.col(column) == 1)["uid"].n_unique(),
                        "auc_users": len(table), "excluded_single_class_users": frame["uid"].n_unique() - len(table),
                        "user_auc": table["auc"].mean(), "ci_low": low, "ci_high": high,
                        "seeds": len(score_columns), "reference_only": len(table) < 500,
                        "display_status": "参考値" if len(table) < 500 else "通常表示"})
    d = tables["D"].rename({"auc": "d_auc", "n": "d_n", "positives": "d_positives"})
    c = tables["C"].rename({"auc": "c_auc", "n": "c_n", "positives": "c_positives"})
    common = d.join(c, on="uid", validate="1:1").sort("uid")
    if not common["d_n"].equals(common["c_n"]):
        raise ValueError("対応付きD/Cのユーザー内ペア数が異なる")
    paired = paired_user_auc_difference(common.select("uid", pl.col("d_auc").alias("auc")),
                                       common.select("uid", pl.col("c_auc").alias("auc")), BOOTSTRAPS)
    difference = {"pairs": len(frame), "d_auc_users": len(d), "c_auc_users": len(c),
                  "paired_pairs": int(common["d_n"].sum()), "d_auc_common": common["d_auc"].mean(),
                  "c_auc_common": common["c_auc"].mean(), "seeds": len(score_columns), **paired,
                  "reference_only": len(common) < 500,
                  "display_status": "参考値" if len(common) < 500 else "通常表示"}
    common = common.with_columns((pl.col("d_auc") - pl.col("c_auc")).alias("difference"))
    return metrics, difference, tables, common


def model_analysis(frame):
    master = frame.filter(~pl.col("prior_like") & pl.col("split").is_in(["validation", "test"]))
    metrics, differences, user_rows = [], [], []
    identity_references = {}
    for filename, specs in SCORE_SPECS:
        resolved = {model: f"{column}_l1" if column == "actr" else column for model, column in specs.items()}
        features = (pl.scan_parquet(config.PROCESSED / filename).select("pair_id", *resolved.values())
                    .join(master.lazy().select("pair_id"), on="pair_id", how="semi").collect())
        scored = master.join(features, on="pair_id", how="left", validate="1:1")
        if scored.height != master.height or any(scored[c].null_count() for c in resolved.values()):
            raise ValueError(f"ペアと保存スコアが一対一に対応しない: {filename}")
        families = sorted({name.split("_seed")[0] for name in resolved})
        for task in ("T1", "T2"):
            for split in ("validation", "test"):
                base = scored.filter((pl.col("task") == task) & (pl.col("split") == split))
                for selection in ("all_pairs", "matching_snapshot"):
                    sub = base if selection == "all_pairs" else base.filter(pl.col("seen_in_training"))
                    for window in WINDOWS:
                        key = {"analysis": "P14", "task": task, "split": split, "selection": selection, "window": window}
                        for family in families:
                            columns = [column for name, column in resolved.items() if name.split("_seed")[0] == family]
                            rows, diff, tables, common = auc_comparison(sub, columns, window)
                            signature = common.select("uid", "d_n", "c_n", "d_positives", "c_positives")
                            identity_key = (task, split, selection, window)
                            if identity_key in identity_references:
                                assert identity_references[identity_key].equals(signature), "モデル間の対応ユーザーが異なる"
                            else:
                                identity_references[identity_key] = signature
                            metrics.extend([{**key, "model": family, **row} for row in rows])
                            differences.append({**key, "model": family, **diff})
                            user_rows.append(common.with_columns([pl.lit(v).alias(k) for k, v in {**key, "model": family}.items()]))
        print(f"P14 completed: {filename}", flush=True)
    return {"p14_auc": pl.DataFrame(metrics), "p14_paired_d_minus_c": pl.DataFrame(differences),
            "p14_paired_user_auc": pl.concat(user_rows)}


def rank_analysis(metrics):
    a = (pl.read_csv(config.RESULTS / "task_a_next_item.csv").filter(pl.col("model").is_in(MODELS))
         .group_by("model").agg(pl.col("ndcg_at_10").mean().alias("A")))
    rows, details = [], []
    for key, group in metrics.group_by("task", "split", "selection", "window", maintain_order=True):
        condition = dict(zip(("task", "split", "selection", "window"), key))
        ranked = group.select("model", "route", "user_auc").pivot(on="route", index="model", values="user_auc").join(a, on="model", how="left", validate="1:1")
        counts = group.select("model", "route", "auc_users").pivot(on="route", index="model", values="auc_users").rename(
            {name: f"{name}_auc_users" for name in ("D", "C", "B")})
        ranked = ranked.join(counts, on="model", validate="1:1")
        for name in ("D", "C", "B", "A"):
            ranked = ranked.with_columns(pl.col(name).rank(method="average", descending=True).alias(f"{name}_rank"))
        for other in ("C", "B", "A"):
            valid = ranked.filter(pl.col("D").is_finite() & pl.col(other).is_finite())
            if len(valid) == 8 and set(valid["model"]) == set(MODELS):
                tau = kendalltau(valid["D"], valid[other], variant="b", method="auto", alternative="two-sided")
                value, p = float(tau.statistic), float(tau.pvalue)
            else:
                value = p = None
            rows.append({**condition, "comparison": f"D_vs_{other}", "models": len(valid), "kendall_tau": value,
                         "p_value": p, "minimum_d_auc_users": int(ranked["D_auc_users"].min()),
                         "reference_only": bool((ranked["D_auc_users"] < 500).any() or
                                                (other in ("C", "B") and (ranked[f"{other}_auc_users"] < 500).any())),
                         "a_reference": "unchanged day149/holdout150-210 NDCG@10" if other == "A" else "same pairs; label-specific AUC users",
                         "interpretation_limit": "8モデルで検出力が低い。非有意から無関係・不一致とは断定しない"})
        details.append(ranked.with_columns([pl.lit(v).alias(k) for k, v in condition.items()]))
    return {"p14_kendall": pl.DataFrame(rows), "p14_rank_details": pl.concat(details)}


def primary_c_context(metrics):
    primary = (pl.read_csv(config.RESULTS / "main_test_results.csv")
               .filter(pl.col("protocol") == "raw_score")
               .select("family", "task", pl.col("user_auc").alias("primary_c_auc"))
               .rename({"family": "model"}))
    current = metrics.filter((pl.col("window") == "like_90d") & (pl.col("split") == "test"))
    current = current.select("task", "selection", "model", "route", "user_auc").pivot(
        on="route", index=["task", "selection", "model"], values="user_auc")
    current = current.join(primary, on=["model", "task"], validate="m:1")
    for source, target in (("primary_c_auc", "primary_c_rank"), ("C", "same_pairs_c_rank"), ("D", "d_rank")):
        current = current.with_columns(pl.col(source).rank(method="average", descending=True)
                                       .over("task", "selection").alias(target))
    return current


def frozen_definition():
    return {
        "classification": "posthoc; recorded before P13/P14 label rates and AUC",
        "P13": "same main L1 cohort; all T1/T2 train/validation/test; prior_like has group precedence",
        "windows_seconds_closed": {name: [0, seconds] for name, seconds in WINDOWS.items()},
        "prior_like": "any observed same uid/item like with timestamp strictly before anchor; not current state",
        "likes_routes": "all; repeated and duplicate events use existence OR",
        "P13_reverse_denominators": ["all C positives/window event including prior", "all C positives/new-like group", "nonprior C positives/new-like group"],
        "P13_within_user": "nonprior users with both like/no-like pairs; equal-user C-rate difference; no both-C-classes condition",
        "P14": "exclude prior; identical D/C/B pairs; unchanged eight raw score families; no fitting or head",
        "P14_evaluation_splits": ["validation", "test"],
        "P14_primary": {"window": "like_90d", "split": "test"},
        "matching_snapshot": "same all-route/all-play-ratio observed-item set: train29/evaluation149; no model-specific vocab",
        "auc": "macro over users with both classes; ties=0.5; seed41/42/43 AUC averaged within user before bootstrap",
        "paired": "D minus C; users with both AUCs defined; same pair counts; seed means before difference",
        "bootstrap": {"unit": "user", "replicates": BOOTSTRAPS, "seed": SEED, "ci": "95% percentile; unadjusted"},
        "ranks": "Kendall tau-b, two-sided method=auto on8 families; C/B label-specific user populations on same pairs",
        "A_reference": "unchanged day149 models, multi-positive NDCG@10 holdout150-210; not refiltered for validation/known tracks",
        "support_gate": "before all new label rates/AUC: T2/test/all-pairs/nonprior/like90 positive users <500 stops",
        "reference_threshold": "each marginal or paired AUC population <500 is reference-only",
        "no_inputs_from_likes": True,
    }


def _protected_files():
    return {str(p): file_hash(p) for p in ROOT.rglob("*") if p.is_file()
            and not any(x in {".git", ".venv", "__pycache__", ".pytest_cache"} for x in p.relative_to(ROOT).parts)
            and OUTPUT not in p.parents}


def _finite_frame(frame):
    floating = [c for c, dtype in frame.schema.items() if dtype in (pl.Float32, pl.Float64)]
    return frame.with_columns([pl.col(c).fill_nan(None) for c in floating])


def _fmt(value, digits=6):
    if value is None or not np.isfinite(value):
        return "算出不能"
    return f"{value:.{digits}f}"


def _interval(row, mean="user_auc", low="ci_low", high="ci_high"):
    return f"{_fmt(row[mean])} [{_fmt(row[low])}, {_fmt(row[high])}]"


def _leaders(frame, column):
    finite = frame.filter(pl.col(column).cast(pl.Float64).is_finite())
    if finite.is_empty():
        return "算出不能"
    return " / ".join(finite.filter(pl.col(column) == finite[column].max())["model"].sort().to_list())


def write_report(tables, audit, gate, execution):
    lines = ["# P13/P14：likeを使う事後追加分析", "",
             "主結果・主指標・既存モデルを変更しない補助分析。両分析はCの結果とRQ3 like層別を見た後に計画した。",
             "", "## 実行前記録とデータ確認", "",
             f"- 実行時のDECISIONS.md:{REGISTRATION['p13_line']}〜246がP13、{REGISTRATION['p14_line']}〜{REGISTRATION['end_line']}がP14。「事後追加／実行前に記録」。記録時の全文はregistered_decisions_snapshot.md。現行の行番号はブランチにより異なる。",
             f"- 記録時刻UTC：{REGISTRATION['recorded_at_utc']}。この追記時点で追加分析のAUC・ラベル別正例率を計算していない（既存C/RQ3の結果は閲覧済み）。",
             f"- likes：{audit['rows']:,}行。列とdtype：`{audit['schema']}`。timestampは秒、5秒刻み、範囲{audit['min_timestamp']}〜{audit['max_timestamp']}。",
             f"- 同じuid×曲に複数イベント：{audit['repeated_user_item_pairs']:,}組。完全一致重複{audit['duplicate_full_rows']:,}行、uid×曲×時刻重複{audit['duplicate_uid_item_timestamp_rows']:,}行。存在ORなので重複の数でラベルは変わらない。",
             "- unlikesは手元になく、取得していない。**取り消しは考慮できない**。既likeは過去イベントを観測した群で、現在のLike状態ではない。",
             "- likesの最終イベントとlistensの最終イベントの差は、ログの収録終了・完全観測を証明しない。今回は既存主コホートを維持し、Like観測の完全性を新たに保証したとは主張しない。",
             f"- 停止判定：既like除外後のT2/test/全ペア/D90正例ユーザー {gate['d_positive_users']:,}人。状態 `{gate['status']}`。", ""]
    if gate["status"] != "ready":
        lines += ["## 継続確認が必要", "", "500人未満のため、P13の率・P14のAUCは計算していない。TMの判断まで停止。", ""]
    else:
        strata, within = tables["p13_strata"], tables["p13_within_user"]
        lines += ["## P13：testのlike層別とユーザー内比較", "",
                  "既likeを優先し、likeあり/なしは既like以外の排他的群。C/Bは[24時間,90日]、like窓は起点から[0,24時間]または[0,90日]（両端含む）。train/validationもCSVに保存。", "",
                  "| task | like窓 | 群 | ペア | ユーザー | C正例率 | B正例率 |", "|---|---|---|---:|---:|---:|---:|"]
        for r in strata.filter(pl.col("split") == "test").iter_rows(named=True):
            group = {"prior_like": "既like", "like": "likeあり", "no_like": "likeなし"}[r["like_group"]]
            lines.append(f"| {r['task']} | {r['window']} | {group} | {r['pairs']:,} | {r['users']:,} | {_fmt(r['c_positive_rate'])} | {_fmt(r['b_positive_rate'])} |")
        lines += ["", "| task | like窓 | 両群を持つユーザー | ユーザー平均C率差（あり−なし）[95%区間] |", "|---|---|---:|---|"]
        for r in within.filter(pl.col("split") == "test").iter_rows(named=True):
            lines.append(f"| {r['task']} | {r['window']} | {r['paired_users']:,} | {_interval(r, 'mean_difference')} |")
        lines += ["", "逆方向の比率はp13_reverse.csv。全C正例/既like除外C正例の分母を分け、既like群の窓内イベントを含む割合と新規like群の割合を別列に保存。",
                  "", "**関連の解釈に限る。** likeした曲はライブラリに入り、自発で再生しやすくなる可能性がある。Yambdaでは自発再生の入口（ライブラリ・検索等）を区別できない。like_24h層のC率が高くても因果効果や『好みの裏付け』とは断定できない。ユーザー内比較は個人差の一部を除くが、曲・時刻等の交絡や逆因果を除かない。", "",
                  "## P14：Dのユーザー内AUCと対応付きD−C", "",
                  "D主ラベルはlike_90d、like_24hは感度分析。全モデル・D/C/Bで同じ既like除外後ペア。AUCはラベルごとに正負両方を持つユーザーを等重み平均。iALS/BPRはユーザー内で3seedのAUCを平均してからbootstrap。",
                  "", "表中のnはAUC対象ユーザー数。n<500は参考値。validationとC/Bの同一ペア上AUCもCSVに保存。対応差はDとC双方のAUCが定義できる共通ユーザーで計算。", ""]
        for window in WINDOWS:
            for selection in ("all_pairs", "matching_snapshot"):
                lines += [f"### {window} / {selection} / test", "",
                          "| モデル | T1 D AUC [95%区間]・n | T2 D AUC [95%区間]・n |", "|---|---|---|"]
                for model in MODELS:
                    cells = []
                    for task in ("T1", "T2"):
                        r = tables["p14_auc"].filter((pl.col("task") == task) & (pl.col("split") == "test") &
                            (pl.col("selection") == selection) & (pl.col("window") == window) &
                            (pl.col("model") == model) & (pl.col("route") == "D")).row(0, named=True)
                        cells.append(f"{_interval(r)}; n={r['auc_users']:,}" + (" **参考値**" if r["reference_only"] else ""))
                    lines.append(f"| {model} | {' | '.join(cells)} |")
                lines += ["", "| モデル | T1 D−C [95%区間]・共通n | T2 D−C [95%区間]・共通n |", "|---|---|---|"]
                for model in MODELS:
                    cells = []
                    for task in ("T1", "T2"):
                        r = tables["p14_paired_d_minus_c"].filter((pl.col("task") == task) & (pl.col("split") == "test") &
                            (pl.col("selection") == selection) & (pl.col("window") == window) & (pl.col("model") == model)).row(0, named=True)
                        cells.append(f"{_interval(r, 'mean_auc_difference', 'difference_ci_low', 'difference_ci_high')}; n={r['paired_users']:,}" +
                                     (" **参考値**" if r["reference_only"] else ""))
                    lines.append(f"| {model} | {' | '.join(cells)} |")
                lines.append("")
        lines += ["## P14：順位の一致（Kendall τ-b、test）", "",
                  "| task | like窓 | 曲集合 | 比較 | モデル数 | τ | p値 | 最小D AUC人数 |", "|---|---|---|---|---:|---:|---:|---:|"]
        for r in tables["p14_kendall"].filter(pl.col("split") == "test").iter_rows(named=True):
            lines.append(f"| {r['task']} | {r['window']} | {r['selection']} | {r['comparison']} | {r['models']} | {_fmt(r['kendall_tau'])} | {_fmt(r['p_value'])} | {r['minimum_d_auc_users']:,}" +
                         ("（参考値）" if r["reference_only"] else "") + " |")
        lines += ["", "8モデルでは検出力が低い。非有意な比較は『一致は確認できなかった』にとどめ、無関係や不一致を証明したとは述べない。区間・p値は多重比較未補正。C/Bは同じペア集合だが、AUC対象ユーザーはラベルごとに異なる。Aは従来の150〜210日holdoutのままで、validation限定・既知曲限定へ再評価していない。", "",
                  "## 主Cの結果との関係", "",
                  "Dと同じペアでのC、および元の主Cの値・順位をp14_primary_c_context.csvに保存。Likeという別ラベルと既like除外によって対象・AUC適格ユーザーが変わるため、主Cの結果の再現失敗や反証とは解釈しない。", ""]
        for task in ("T1", "T2"):
            context = tables["p14_primary_c_context"].filter((pl.col("task") == task) & (pl.col("selection") == "all_pairs"))
            primary_top = _leaders(context, "primary_c_auc")
            d_top = _leaders(context, "D")
            c_top = _leaders(context, "C")
            lines.append(f"- {task}：元の主Cの首位は{primary_top}。既like除外後の同一ペアCの首位は{c_top}、D90の首位は{d_top}（記述的順位。優越性を検定した表ではない）。")
        lines += ["", "## 言えること／言えないこと", "",
                  "- 言えること：この観測ログ・固定コホートでのlikeイベントとC/Bの関連、保存済み生スコアによる補助ラベルDの識別性能、共通ユーザーでのD−C差。",
                  "- 言えないこと：likeがCを増やした因果効果、likeが潜在的な好みを独立に検証したという主張、ライブラリ等の入口別効果、Like取消を含む現在状態、Like収録の完全性、Dでの結果による主Cのモデル選び直し。",
                  "- 参考値：各条件のAUC人数・対応人数が500未満のときに表示。支持人数や適格ユーザー構成の差を無視してモデル順位を一般化しない。", ""]
    lines += ["## 再現・保護", "",
              "- 再現入口：notebooks/07_like_analysis.ipynb。共通処理：src/relisten/like_analysis.py。",
              "- ラベル境界・経路違い重複・既like優先・空/未来・seed平均/対応差の手作りテスト：tests/test_like_analysis.py。",
              f"- 実行時間：{execution['elapsed_seconds']:.2f}秒。既存保護ファイル{execution['protected_files']:,}件のSHA256一致。DECISIONS.mdの既存部分は別途prefix hashで保護。",
              "- ソース・入力のSHA256と実行前定義：protocol.json。CSVの列は条件を明示し、既存の結果へ混ぜない。", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


def run_like_analysis():
    """実行前記録確認→ラベル→支持人数gate→P13→P14の順で実行する。"""
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = _protected_files()
    decisions = ROOT / "DECISIONS.md"
    verify_registration(decisions, OUTPUT / "registered_decisions_snapshot.md")
    audit = audit_likes()
    input_paths = [config.PROCESSED / "pairs.parquet", config.PROCESSED / "seen_item_diagnostics/pair_flags.parquet",
                   config.RAW / "flat/50m/likes.parquet", config.RESULTS / "task_a_next_item.csv",
                   config.RESULTS / "main_test_results.csv", *[config.PROCESSED / filename for filename, _ in SCORE_SPECS]]
    protocol = {"registration": REGISTRATION, "definition": frozen_definition(), "data_audit": audit,
                "input_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in input_paths},
                "like_observation_limit": "last like event is not proof of collection end/completeness; original cohort retained"}
    path = OUTPUT / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["definition"] != protocol["definition"] or old["input_hashes"] != protocol["input_hashes"]:
            raise ValueError("既存like分析の定義または入力が違う。元定義を消さずDECISIONS.mdに変更理由を追記する")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print("実行前記録・データ監査を確認。Likeのラベルを作ります。", flush=True)
    frame = load_cohort()
    gate = support_gate(frame)
    pl.DataFrame([gate]).write_csv(OUTPUT / "support_gate.csv")
    print("支持人数gate:", gate, flush=True)
    tables = {}
    if gate["status"] == "ready":
        tables.update(descriptive_analysis(frame))
        print("P13 completed", flush=True)
        tables.update(model_analysis(frame))
        tables.update(rank_analysis(tables["p14_auc"]))
        tables["p14_primary_c_context"] = primary_c_context(tables["p14_auc"])
        for name, table in tables.items():
            _finite_frame(table).write_csv(OUTPUT / f"{name}.csv")
    after = _protected_files()
    if after != protected:
        raise ValueError("既存ファイルの変更または許可されていない追加を検出")
    execution = {"elapsed_seconds": time.monotonic() - started, "protected_files": len(protected),
                 "changed_protected_files": [], "status": gate["status"],
                 "source_sha256": file_hash(Path(__file__)), "metrics_sha256": file_hash(ROOT / "src/relisten/metrics.py")}
    protocol.update({"support_gate": gate, "execution": execution})
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    write_report(tables, audit, gate, execution)
    return {"status": gate["status"], "support_gate": gate, "audit": audit, "tables": tables, "execution": execution}
