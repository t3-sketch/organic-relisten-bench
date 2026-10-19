"""ユーザーを等重みとするAUCと、共通の確率評価。"""
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .config import SEED


def user_auc_table(uid, label, score):
    if not np.isin(label, [0, 1]).all() or not np.isfinite(np.asarray(score)).all():
        raise ValueError("二値ラベルと有限のスコアが必要")
    frame = pl.DataFrame({"uid": uid, "label": np.asarray(label, dtype=np.uint8), "score": score})
    ranked = frame.with_columns(pl.col("score").rank(method="average").over("uid").alias("rank"))
    return ranked.group_by("uid").agg(
        pl.len().alias("n"), pl.col("label").sum().cast(pl.Int64).alias("positives"),
        (pl.col("rank")*pl.col("label")).sum().alias("positive_rank_sum"),
    ).filter((pl.col("positives")>0)&(pl.col("positives")<pl.col("n"))).with_columns(
        ((pl.col("positive_rank_sum")-pl.col("positives")*(pl.col("positives")+1)/2)
         /(pl.col("positives")*(pl.col("n")-pl.col("positives")))).alias("auc")
    ).select("uid", "n", "positives", "auc").sort("uid")


def bootstrap_mean(values, n_bootstrap=1000, seed=SEED):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)])
    return tuple(np.quantile(means, [.025, .975]))


def evaluate_ranking(uid, label, score, n_bootstrap=1000):
    y, p = np.asarray(label), np.asarray(score, dtype=float)
    if len(uid)!=len(y) or len(y)!=len(p) or not len(y) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
        raise ValueError("同じ長さの非空uid・二値ラベル・有限スコアが必要")
    user_auc = user_auc_table(uid, y, p)
    low, high = bootstrap_mean(user_auc["auc"], n_bootstrap)
    result = {"pairs": len(y), "users": len(np.unique(uid)), "auc_users": user_auc.height,
              "positive_rate": float(y.mean()), "user_auc": user_auc["auc"].mean(),
              "user_auc_ci_low": low, "user_auc_ci_high": high,
              "global_auc": float(roc_auc_score(y,p)) if np.unique(y).size==2 else float("nan"),
              "pr_auc": float(average_precision_score(y,p)) if y.any() else float("nan")}
    return result, user_auc


def evaluate_probabilities(uid, label, probability, n_bootstrap=1000):
    p=np.asarray(probability,dtype=float)
    if np.any((p<0)|(p>1)):raise ValueError("0〜1の確率が必要")
    result,auc=evaluate_ranking(uid,label,p,n_bootstrap)
    result.update(log_loss=float(log_loss(label,p,labels=[0,1])),brier=float(brier_score_loss(label,p)))
    return result,auc


def paired_user_auc_difference(left, right, n_bootstrap=1000):
    joined = left.select("uid", pl.col("auc").alias("left_auc")).join(
        right.select("uid", pl.col("auc").alias("right_auc")), on="uid", how="inner").sort("uid")
    differences = (joined["left_auc"]-joined["right_auc"]).to_numpy()
    low, high = bootstrap_mean(differences, n_bootstrap)
    return {"paired_users": len(differences), "mean_auc_difference": float(differences.mean()) if len(differences) else float("nan"),
            "difference_ci_low": low, "difference_ci_high": high}
