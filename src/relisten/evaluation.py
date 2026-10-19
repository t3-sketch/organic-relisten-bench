"""同じラベル・同じヘッドで各モデルを測る。"""
import json
import warnings

import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .labels import with_labels
from .light_models import benchmark_pairs
from .metrics import evaluate_probabilities, paired_user_auc_difference


BASELINE_COLUMNS=["past_return_rate","past_active_days"]
LIGHT_MODELS={"MostPop":"mostpop","DecayPop":"decaypop","AudioCosine":"audio_cosine","ACT-R":"actr"}


def common_head(train_features,train_labels,evaluation_features):
    if np.unique(train_labels).size!=2:
        raise ValueError("共通ヘッドの学習には正例と負例が必要")
    if not np.isfinite(train_features).all() or not np.isfinite(evaluation_features).all():
        raise ValueError("ヘッド入力に欠損・無限値がある")
    head=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,tol=1e-7,
                                                        random_state=config.SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        head.fit(train_features,train_labels)
    probability=head.predict_proba(evaluation_features)[:,1]
    return probability,head


def evaluate_features(features,models,filename="light_model_metrics.csv",tasks=("T2","T1"),artifact_root=None):
    """L=1/3/7/14、T1/T2、B/Cにそれぞれ同じヘッドを再学習する。"""
    pairs=benchmark_pairs().filter(pl.col("task").is_in(tasks))
    artifact_root=config.PROCESSED if artifact_root is None else artifact_root
    frame=pairs.join(features.select(pl.all().exclude("uid","item_id","anchor_timestamp","task","split")),on="pair_id")
    if frame.height!=pairs.height or frame["pair_id"].n_unique()!=pairs.height:
        raise ValueError("起点と特徴の一対一対応が崩れている")
    expected_day=pl.when(pl.col("split")=="train").then(config.HEAD_REPRESENTATION_END_DAY).otherwise(config.REPRESENTATION_END_DAY)
    assert frame.select((pl.col("model_end_day")==expected_day).all()).item()
    assert frame.select((pl.col("anchor_timestamp")>=(pl.col("model_end_day")+1)*config.SECONDS_PER_DAY).all()).item()
    metrics,paired=[],[]
    # 同一L・経路・タスクで全モデルの対象ユーザーを揃える。
    for lower in sorted(set(config.SENSITIVITY_LOWER_DAYS)|{config.LOWER_BOUND_DAYS}):
        eligible=with_labels(frame,lower)
        for task in tasks:
            cohort=eligible.filter(pl.col("task")==task)
            train=cohort.filter(pl.col("split")=="train")
            for route in ("c","b"):
                y_train=train[f"label_{route}"].to_numpy()
                baseline_auc={}
                specs=[("UserBaseline","baseline",BASELINE_COLUMNS)]
                for name,column in models.items():
                    score=f"{column}_l{lower}" if f"{column}_l{lower}" in features.columns else column
                    specs.extend([(name,"score_only",[score]),(name,"baseline_plus_score",BASELINE_COLUMNS+[score])])
                for name,protocol,columns in specs:
                    x_train=train.select(columns).to_numpy()
                    head=None
                    for split in ("validation","test"):
                        evaluation=cohort.filter(pl.col("split")==split)
                        x_eval=evaluation.select(columns).to_numpy()
                        if head is None:
                            probability,head=common_head(x_train,y_train,x_eval)
                        else:
                            probability=head.predict_proba(x_eval)[:,1]
                        result,auc=evaluate_probabilities(evaluation["uid"].to_numpy(),evaluation[f"label_{route}"].to_numpy(),probability)
                        key={"model":name,"protocol":protocol,"task":task,"route":route.upper(),"split":split,
                             "lower_days":lower,"is_main_lower":lower==config.LOWER_BOUND_DAYS}
                        metrics.append({**key,**result})
                        auc_directory=artifact_root/"user_auc"
                        auc_directory.mkdir(parents=True,exist_ok=True)
                        auc.write_parquet(auc_directory/f"{name}_{protocol}_{task}_{route}_{split}_l{lower}.parquet")
                        if name=="UserBaseline":baseline_auc[split]=auc
                        else:paired.append({**key,**paired_user_auc_difference(auc,baseline_auc[split])})
                        # 後のRQ3とモデル間の対応付き比較用。主Cのみ確率を保存する。
                        if lower==config.LOWER_BOUND_DAYS and route=="c":
                            directory=artifact_root/"predictions"
                            directory.mkdir(parents=True,exist_ok=True)
                            evaluation.select("pair_id","uid","item_id","anchor_timestamp","task","split","label_c").with_columns(
                                pl.Series("probability",probability)).write_parquet(directory/f"{name}_{protocol}_{task}_{split}.parquet")
                print(f"evaluated L={lower} {task} {route.upper()}",flush=True)
    table=pl.DataFrame(metrics)
    assert {config.LOWER_BOUND_DAYS,7,14}.issubset(set(table["lower_days"]))
    table.write_csv(config.RESULTS/filename)
    pl.DataFrame(paired).write_csv(config.RESULTS/filename.replace("metrics","paired_vs_user_baseline"))
    protocol_name="common_head_protocol.json" if artifact_root==config.PROCESSED else filename.removesuffix(".csv")+"_head_protocol.json"
    (config.RESULTS/protocol_name).write_text(json.dumps({
        "training_labels":"train only","scaler_fit":"train only","C":1,"solver":"lbfgs","max_iter":1000,
        "class_weight":None,"tuning_on_validation":False,"calibration":"shared logistic (Platt) head",
        "pr_auc_definition":"average_precision","user_auc":"macro over users with both classes; ties use average ranks",
        "bootstrap_unit":"user","bootstrap_replicates":1000,"seed":config.SEED,
        "head_model_cutoff_day":config.HEAD_REPRESENTATION_END_DAY,"eval_model_cutoff_day":config.REPRESENTATION_END_DAY,
        "required_sensitivity_lower_days":[7,14],"baseline_columns":BASELINE_COLUMNS,
    },indent=2)+"\n")
    return table
