"""L の候補は学習用の主タスクだけで選び、その後に評価期間を記述する。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import config
from .labels import with_labels


FIGURES = config.RESULTS / "figures"
COLORS = {"C": "#18747e", "B": "#d9843b", "O2": "#7161a3"}


def return_curve(pairs, route="c"):
    first = pairs[f"first_relisten_seconds_{route}"].drop_nulls().to_numpy()
    elapsed_day = np.ceil(first / config.SECONDS_PER_DAY).astype(int)
    counts = np.bincount(elapsed_day, minlength=config.HORIZON_DAYS + 1)[:config.HORIZON_DAYS + 1]
    return pl.DataFrame({
        "day": np.arange(config.HORIZON_DAYS + 1), "first_returns": counts,
        "at_risk_at_origin": np.full(config.HORIZON_DAYS + 1, pairs.height),
        "daily_rate": counts / pairs.height if pairs.height else np.full(len(counts), np.nan),
        "cumulative_rate": np.cumsum(counts) / pairs.height if pairs.height else np.full(len(counts), np.nan),
    })


def recommend_lower_bound(training_pairs):
    if set(training_pairs["split"].unique()) != {"train"}:
        raise ValueError("L の選定に検証・テストを使えない")
    if set(training_pairs["task"].unique()) != {"T2"}:
        raise ValueError("主タスクT2で候補を選定する")
    if not training_pairs["complete_followup_90"].all():
        raise ValueError("候補選定には90日を観測できた起点だけを使う")
    curve = return_curve(training_pairs)
    mean_tail = curve.filter(pl.col("day").is_between(30, config.HORIZON_DAYS))["daily_rate"].mean()
    eligible = curve.filter((pl.col("day") >= 1) & (pl.col("daily_rate") <= 2 * mean_tail))
    crossing = int(eligible["day"][0]) if eligible.height else None
    candidate = min(config.SENSITIVITY_LOWER_DAYS, key=lambda day: (abs(day-crossing), day)) if crossing else None
    return curve, {"task": "T2", "split": "train", "cohort": "all complete-followup anchors; no activity selection",
                   "pairs": training_pairs.height, "tail_daily_mean": mean_tail,
                   "threshold": 2*mean_tail, "crossing_day": crossing,
                   "candidate_lower_days": candidate, "configured_lower_days": config.LOWER_BOUND_DAYS,
                   "approved": False, "rounding": "nearest among 1,3,7,14; smaller value on ties"}


def plot_curves(pairs, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    reference = pairs.filter(pl.col("task") == "O2")
    for col, task in enumerate(("T2", "T1")):
        cohort = pairs.filter(pl.col("task") == task)
        for route, label in (("c", "C: organic return"), ("b", "B: any-route return")):
            curve = return_curve(cohort, route)
            for row, metric in enumerate(("cumulative_rate", "daily_rate")):
                axes[row, col].plot(curve["day"], curve[metric] * 100, label=label, color=COLORS[route.upper()])
        curve = return_curve(reference)
        for row, metric in enumerate(("cumulative_rate", "daily_rate")):
            axes[row, col].plot(curve["day"], curve[metric]*100, label="Organic first contact: C", color=COLORS["O2"], linestyle="--")
            axes[row, col].set_title(f"{task} — {'cumulative' if row == 0 else 'first return on day'}")
            axes[row, col].set_ylabel("Fraction of original cohort (%)")
            axes[row, col].grid(alpha=.18)
        axes[1, col].set_xlabel("Elapsed day after anchor")
        axes[0, col].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=170)
    plt.close(fig)


def training_eda():
    FIGURES.mkdir(parents=True, exist_ok=True)
    # この関数では評価用ペアを実体化しない。
    train = pl.scan_parquet(config.PROCESSED / "pairs.parquet").filter(
        (pl.col("split") == "train") & pl.col("complete_followup_90")
    ).collect()
    curve, candidate = recommend_lower_bound(train.filter(pl.col("task") == "T2"))
    curve.write_csv(config.RESULTS / "train_t2_lower_bound_curve.csv")
    (config.RESULTS / "lower_bound_candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    plot_curves(train, "train_return_curves.png", "Training anchors only — all activity levels")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curve["day"], curve["daily_rate"]*100, color=COLORS["C"], label="T2 training: first organic return")
    ax.axhline(candidate["threshold"]*100, color="#b14b4b", linestyle="--", label="2 × mean on days 30–90")
    if candidate["crossing_day"] is not None:
        ax.axvline(candidate["crossing_day"], color="#666666", linestyle=":", label=f"First crossing: day {candidate['crossing_day']}")
    ax.set(xlabel="Elapsed day after anchor", ylabel="Fraction of original cohort (%)", title="Lower-bound proposal — training T2 only")
    ax.legend(fontsize=8)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "lower_bound_selection.png", dpi=170)
    plt.close(fig)
    return candidate


def descriptive_eda():
    candidate_path = config.RESULTS / "lower_bound_candidate.json"
    if not candidate_path.exists():
        raise RuntimeError("評価期間を見る前に training_eda() で候補を記録する")
    candidate = json.loads(candidate_path.read_text())
    pairs = pl.read_parquet(config.PROCESSED / "pairs.parquet")
    complete = pairs.filter(pl.col("complete_followup_90"))
    tables, early = [], []
    for split in config.SPLITS:
        split_pairs = complete.filter(pl.col("split") == split)
        plot_curves(split_pairs, f"{split}_return_curves.png", f"{split.capitalize()} anchors — descriptive check after L proposal")
        plot_curves(with_labels(split_pairs), f"{split}_active30_return_curves.png",
                    f"{split.capitalize()} anchors — active >=30 days in configured window")
        for task in ("T2", "T1", "O2"):
            subset = split_pairs.filter(pl.col("task") == task)
            for route in ("b", "c"):
                tables.append(return_curve(subset, route).with_columns(
                    pl.lit(split).alias("split"), pl.lit(task).alias("task"), pl.lit(route.upper()).alias("route")))
                first = subset[f"first_relisten_seconds_{route}"]
                early.append({"split": split, "task": task, "route": route.upper(), "pairs": subset.height,
                              "within_24h": int((first <= config.SECONDS_PER_DAY).fill_null(False).sum())})
    pl.concat(tables).write_csv(config.RESULTS / "first_return_curves.csv")
    pl.DataFrame(early).with_columns((pl.col("within_24h")/pl.col("pairs")).alias("rate")).write_csv(
        config.RESULTS / "returns_within_24h.csv")
    _stratified_curves(complete.filter(pl.col("split") == "train"))
    _basic_tables(pairs)
    sensitivity = pl.read_csv(config.RESULTS / "label_sensitivity.csv").filter(
        (pl.col("task") == "T2") & (pl.col("horizon_days") == config.HORIZON_DAYS) & (pl.col("min_active_days") == 30))
    sensitivity.write_csv(config.RESULTS / "candidate_rate_comparison.csv")
    _write_report(candidate, sensitivity)
    return candidate, sensitivity


def _stratified_curves(train):
    warmup = pl.scan_parquet(config.RAW / "flat/50m/listens.parquet").filter(
        pl.col("timestamp") < min(lo for lo, hi in config.SPLITS.values()) * config.SECONDS_PER_DAY
    ).group_by("item_id").len(name="warmup_popularity").collect()
    train = train.join(warmup, on="item_id", how="left").with_columns(pl.col("warmup_popularity").fill_null(0))
    pop_median = train.filter(pl.col("warmup_popularity") > 0)["warmup_popularity"].median()
    activity_column = f"active_days_l{config.LOWER_BOUND_DAYS}_h{config.HORIZON_DAYS}"
    activity_cuts = np.unique(np.quantile(train[activity_column].to_numpy(), [.25, .5, .75]))
    train = train.with_columns(
        pl.Series("activity_group", [f"Q{i+1}" for i in np.searchsorted(activity_cuts, train[activity_column].to_numpy(), side="left")]),
        pl.when(pl.col("warmup_popularity")==0).then(pl.lit("unseen in warm-up"))
        .when(pl.col("warmup_popularity")<=pop_median).then(pl.lit("lower popularity"))
        .otherwise(pl.lit("higher popularity")).alias("popularity_group"),
    )
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, task in enumerate(("T2", "T1")):
        cohort = train.filter(pl.col("task")==task)
        for row, column in enumerate(("activity_group", "popularity_group")):
            for group in sorted(cohort[column].unique()):
                sub = cohort.filter(pl.col(column)==group)
                curve = return_curve(sub)
                axes[row, col].plot(curve["day"], curve["cumulative_rate"]*100, label=f"{group} (n={sub.height:,})")
                rows.append(curve.with_columns(pl.lit(task).alias("task"), pl.lit(column).alias("stratifier"), pl.lit(group).alias("stratum")))
            axes[row, col].set_title(f"{task} — {column.replace('_', ' ')}")
            axes[row, col].legend(fontsize=7)
            axes[row, col].set_ylabel("Cumulative organic return (%)")
            axes[row, col].grid(alpha=.18)
        axes[1, col].set_xlabel("Elapsed day after anchor")
    fig.suptitle("Training only — popularity measured before day 30")
    fig.tight_layout()
    fig.savefig(FIGURES / "train_stratified_curves.png", dpi=170)
    plt.close(fig)
    pl.concat(rows).write_csv(config.RESULTS / "train_stratified_curves.csv")
    (config.RESULTS / "strata_definition.json").write_text(json.dumps({
        "activity_quantiles": activity_cuts.tolist(), "activity_window": f"[{config.LOWER_BOUND_DAYS},90]",
        "popularity_period": "0..29 days", "positive_popularity_median": pop_median,
    }, indent=2) + "\n")


def _basic_tables(pairs):
    listens = pl.scan_parquet(config.RAW / "flat/50m/listens.parquet")
    quantiles = [.0, .1, .25, .5, .75, .9, .99, 1.0]
    listens.select([pl.col("played_ratio_pct").quantile(q).alias(f"q{q:g}") for q in quantiles]).collect().write_csv(
        config.RESULTS / "played_ratio_quantiles.csv")
    listens.select(pl.col("played_ratio_pct").cut([25,50,75,100,125,150,200,500,1000]).alias("played_ratio_band")).group_by(
        "played_ratio_band").len().collect().sort("played_ratio_band").write_csv(config.RESULTS / "played_ratio_distribution.csv")
    activity = pl.scan_parquet(config.PROCESSED / "user_days.parquet").group_by("uid").agg(pl.len().alias("active_days")).collect()
    activity.write_csv(config.RESULTS / "user_activity_days.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(activity["active_days"], bins=30, color=COLORS["C"], edgecolor="white")
    ax.set(xlabel="Observed active days", ylabel="Users", title="User activity across the released listening log")
    fig.tight_layout()
    fig.savefig(FIGURES / "user_activity_days.png", dpi=170)
    plt.close(fig)
    known = pl.scan_parquet(config.PROCESSED / "representation_train.parquet").select("item_id").unique().collect().with_columns(pl.lit(True).alias("in_representation_train"))
    embedded = pl.read_parquet(config.PROCESSED / "embedding_item_ids_50m.parquet").with_columns(pl.lit(True).alias("has_embedding"))
    t2 = pairs.filter(pl.col("task")=="T2").join(known, on="item_id", how="left").join(embedded, on="item_id", how="left").with_columns(
        pl.col("in_representation_train").fill_null(False), pl.col("has_embedding").fill_null(False))
    t2.group_by("split").agg(
        pl.len().alias("pairs"), (~pl.col("in_representation_train")).mean().alias("cold_pair_fraction"),
        pl.col("has_embedding").mean().alias("embedded_pair_fraction"),
    ).write_csv(config.RESULTS / "t2_cold_item_pairs.csv")
    t2.unique(subset=["split", "item_id"]).group_by("split").agg(
        pl.len().alias("unique_items"), (~pl.col("in_representation_train")).mean().alias("cold_item_fraction"),
        pl.col("has_embedding").mean().alias("embedded_item_fraction"),
    ).write_csv(config.RESULTS / "t2_cold_items.csv")


def _write_report(candidate, sensitivity):
    summary = pl.read_csv(config.RESULTS / "split_summary.csv")
    protocol_path = config.RESULTS / "label_protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    lines = ["# マイルストーン1・2の結果", "", f"## 採用L={config.LOWER_BOUND_DAYS}でのラベル表", "",
             "| タスク | 分割 | ペア数 | C正例率 | B正例率 |", "|---|---|---:|---:|---:|"]
    for row in summary.iter_rows(named=True):
        lines.append(f"| {row['task']} | {row['split']} | {row['pairs']:,} | {row['positive_rate_c']:.3%} | {row['positive_rate_b']:.3%} |")
    lines += ["", "## EDA規則の候補（採用値とは区別）", "",
              f"学習用T2のみ（活動条件なし、90日追跡可能な{candidate['pairs']:,}ペア）で選定。",
              f"30〜90日目の平均の2倍は{candidate['threshold']:.6%}/日。最初にこの値以下になるのは{candidate['crossing_day']}日目。",
              f"EDAの自動規則による候補は **L = {candidate['candidate_lower_days']}**。採用した設定値は **{config.LOWER_BOUND_DAYS}**。",
              "評価期間の曲線は候補ファイル作成後に確認した。候補選択には使っていない。", "",
              "| L | 分割 | ペア数（30日以上活動） | C正例率 |", "|---:|---|---:|---:|"]
    for row in sensitivity.sort("lower_days", "split").iter_rows(named=True):
        lines.append(f"| {row['lower_days']} | {row['split']} | {row['pairs']:,} | {row['positive_rate_c']:.3%} |")
    lines += ["", "## 根拠と再現", "", "- `figures/lower_bound_selection.png`：学習T2での選定基準。",
              "- `figures/train_return_curves.png`：T1/T2、B/C、自発初接触の曲線。",
              "- `figures/train_stratified_curves.png`：利用日数分位と過去の人気度による違い。",
              "- `first_return_curves.csv`、`returns_within_24h.csv`：各分割の再聴時間。",
              "- `label_sensitivity.csv`：L×60/90日×活動条件0/10/30日の全表。",
              "- `dataset_summary.csv`、`embedding_coverage.csv`、`t2_cold_items.csv`：基本統計とカバレッジ。",
              "- `../DECISIONS.md`：timestamp単位とT1の解釈の根拠。", "",
              (f"**TMによる承認済み：L={protocol.get('lower_days')}。窓の両端を含む。** 理由：{protocol.get('reason')}"
               if protocol.get("approved") else "**ここで停止。TMがLを決めるまでマイルストーン3は実行しない。**"), ""]
    (config.RESULTS / "MILESTONE_REPORT.md").write_text("\n".join(lines))
