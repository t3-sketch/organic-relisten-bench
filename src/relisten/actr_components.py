"""結果確認後のT2診断。主ACT-Rを変更せずS/Pとorganic入力を分ける。"""
import json
import fcntl

import numpy as np
import polars as pl
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from . import config
from .evaluation import evaluate_features,common_head
from .light_models import _audio_arrays,benchmark_pairs,fit_item_sessions


DIAGNOSTIC_MODELS={"ACT-R":"actr","AudioCosine":"audio_cosine",
    "ACT-R-S-only":"s_only","ACT-R-P-only":"p_only","ACT-R-organic-SP":"organic_sp",
    "ACT-R-S-raw":"s_raw","ACT-R-P-raw":"p_raw","ACT-R-candidate-count":"candidate_count"}


def _spreading_lookup(local,contexts,model):
    trained_items,sessions,prevalence=model
    ti=np.searchsorted(trained_items,local)
    tm=(ti<len(trained_items))&(trained_items[np.minimum(ti,len(trained_items)-1)]==local)
    tc=np.searchsorted(trained_items,contexts)
    tcm=(tc<len(trained_items))&(trained_items[np.minimum(tc,len(trained_items)-1)]==contexts)
    spreading=np.zeros((len(local),len(contexts)),dtype=np.float32)
    if tm.any() and tcm.any():
        co=(sessions[ti[tm]]@sessions[tc[tcm]].T).toarray()
        spreading[np.ix_(tm,tcm)]=co*sessions.shape[1]/(prevalence[ti[tm],None]*prevalence[tc[tcm]][None,:])
    return spreading


def _lookups(local,contexts,audio_items,audio,model):
    ai=np.searchsorted(audio_items,local)
    available=(ai<len(audio_items))&(audio_items[np.minimum(ai,len(audio_items)-1)]==local)
    vectors=np.zeros((len(local),audio.shape[1]),dtype=np.float32)
    vectors[available]=audio[ai[available]]
    ci=np.searchsorted(local,contexts)
    cmask=(ci<len(local))&(local[np.minimum(ci,len(local)-1)]==contexts)
    context_vectors=np.zeros((len(contexts),audio.shape[1]),dtype=np.float32)
    context_vectors[cmask]=vectors[ci[cmask]]
    partial=vectors@context_vectors.T
    return _spreading_lookup(local,contexts,model),partial,available


def _user_components(pairs,history,audio_items,audio,all_model,organic_model):
    times=history["timestamp"].to_numpy()
    items=history["item_id"].to_numpy()
    organic=history["is_organic"].to_numpy()==1
    anchors=pairs["anchor_timestamp"].to_numpy()
    ends=np.searchsorted(times,anchors,side="left")
    org_times,org_ids=times[organic],items[organic]
    org_ends=np.searchsorted(org_times,anchors,side="left")
    context=np.where(ends>0,items[np.maximum(ends-1,0)],0)
    org_context=np.array([org_ids[e-1] if e else 0 for e in org_ends])
    contexts=np.unique(np.r_[context,org_context])
    local=np.unique(np.r_[items,pairs["item_id"].to_numpy()])
    index=np.searchsorted(local,items)
    org_index=index[organic]
    target_index=np.searchsorted(local,pairs["item_id"].to_numpy())
    spreading,partial,available=_lookups(local,contexts,audio_items,audio,all_model)
    org_spreading=_spreading_lookup(local,contexts,organic_model)
    rows=[]
    for j,(end,org_end,target) in enumerate(zip(ends,org_ends,target_index)):
        candidates=np.unique(np.r_[index[:end],target])
        position=np.searchsorted(candidates,target)
        col=np.searchsorted(contexts,context[j])
        s=softmax(spreading[candidates,col])
        p=softmax(partial[candidates,col])
        org_candidates=np.unique(np.r_[org_index[:org_end],target])
        org_position=np.searchsorted(org_candidates,target)
        org_col=np.searchsorted(contexts,org_context[j])
        # 文脈なしは全成分0の一様softmax。主版と同じ欠損時の規則を明示する。
        os=org_spreading[org_candidates,org_col] if org_end else np.zeros(len(org_candidates))
        op=partial[org_candidates,org_col] if org_end else np.zeros(len(org_candidates))
        self_position=np.searchsorted(candidates,np.searchsorted(local,context[j])) if end else None
        context_index=index[end-1] if end else 0
        rows.append({"pair_id":pairs["pair_id"][j],"s_only":float(s[position]),"p_only":float(p[position]),
            "organic_sp":float(softmax(os)[org_position]+softmax(op)[org_position]),
            "s_raw":float(spreading[target,col]),"p_raw":float(partial[target,col]),
            "candidate_count":len(candidates),"organic_candidate_count":len(org_candidates),
            "s_max_mass":float(s.max()),"s_self_mass":float(s[self_position]) if end else 0.,
            "context_available":bool(end),"context_is_organic":bool(organic[end-1]) if end else False,
            "partial_audio_available":bool(end and available[target] and available[context_index]),
            "organic_context_available":bool(org_end)})
    return pl.DataFrame(rows)


def build_component_features():
    path=config.PROCESSED/"actr_component_features.parquet"
    if path.exists():return pl.read_parquet(path)
    pairs=benchmark_pairs().filter(pl.col("task")=="T2")
    original=pl.read_parquet(config.PROCESSED/"light_features.parquet")
    audio_items,audio=_audio_arrays()
    history=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").select(
        "uid","item_id",pl.col("timestamp").cast(pl.Int64),"is_organic").filter(
        pl.col("timestamp")<=pairs["anchor_timestamp"].max()).sort("uid","timestamp","item_id").collect()
    histories={h["uid"][0]:h for h in history.partition_by("uid",maintain_order=True)}
    rows=[]
    for day in (config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY):
        selected=pairs.filter(pl.col("split")=="train" if day==config.HEAD_REPRESENTATION_END_DAY else pl.col("split")!="train")
        all_items,all_sessions=fit_item_sessions(day)
        org_items,org_sessions=fit_item_sessions(day,organic_only=True)
        all_model=(all_items,all_sessions,np.asarray(all_sessions.sum(axis=1)).ravel())
        organic_model=(org_items,org_sessions,np.asarray(org_sessions.sum(axis=1)).ravel())
        groups=selected.partition_by("uid",maintain_order=True)
        with threadpool_limits(limits=4):
            for number,group in enumerate(groups):
                rows.append(_user_components(group,histories[group["uid"][0]],audio_items,audio,
                    all_model,organic_model))
                if number%500==0:print(f"T2 ACT-R components day{day}: {number}/{len(groups)} users",flush=True)
        del all_sessions,org_sessions,all_model,organic_model
    result=pl.concat(rows).join(original.select("pair_id","model_end_day","past_return_rate","past_active_days",
        "audio_cosine",*[f"actr_l{x}" for x in config.SENSITIVITY_LOWER_DAYS]),on="pair_id").sort("pair_id")
    for lower in config.SENSITIVITY_LOWER_DAYS:
        assert np.allclose(result["s_only"]+result["p_only"],result[f"actr_l{lower}"],rtol=1e-5,atol=1e-7),"S+P differs from original T2 ACT-R"
    result.write_parquet(path)
    return result


def run_component_analysis():
    with (config.PROCESSED/"actr_components.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result=_run_component_analysis()
        if not (config.RESULTS/"actr_component_head_directions.csv").exists():
            _head_direction_diagnostics()
        if not (config.RESULTS/"actr_component_mechanism_differences.csv").exists():
            _component_mechanism_differences()
        return result


def _component_mechanism_differences():
    from .metrics import paired_user_auc_difference
    directory=config.PROCESSED/"actr_diagnostics/user_auc"
    comparisons=[("ACT-R","ACT-R-P-only"),("ACT-R-S-raw","ACT-R-S-only"),
        ("ACT-R-P-only","ACT-R-P-raw"),("ACT-R-candidate-count","ACT-R")]
    rows=[]
    for lower in config.SENSITIVITY_LOWER_DAYS:
        for left,right in comparisons:
            tables=[pl.read_parquet(directory/f"{model}_score_only_T2_c_test_l{lower}.parquet").sort("uid")
                    for model in (left,right)]
            assert tables[0].select("uid","n","positives").equals(tables[1].select("uid","n","positives")),"Diagnostic cohorts differ"
            rows.append({"left_model":left,"right_model":right,"lower_days":lower,"task":"T2",
                "route":"C","split":"test","protocol":"score_only",**paired_user_auc_difference(*tables)})
    pl.DataFrame(rows).write_csv(config.RESULTS/"actr_component_mechanism_differences.csv")


def _head_direction_diagnostics():
    """較正前の向きと、学習だけで選ばれたヘッドの向きを分けて記録する。"""
    from sklearn.metrics import roc_auc_score
    from .labels import with_labels
    from .metrics import user_auc_table
    features=pl.read_parquet(config.PROCESSED/"actr_component_features.parquet")
    cohort=with_labels(benchmark_pairs().filter(pl.col("task")=="T2").join(features,on="pair_id"))
    train=cohort.filter(pl.col("split")=="train")
    evaluation=cohort.filter(pl.col("split")=="test")
    rows=[]
    for model,column in DIAGNOSTIC_MODELS.items():
        column=f"actr_l{config.LOWER_BOUND_DAYS}" if column=="actr" else column
        probability,head=common_head(train.select(column).to_numpy(),train["label_c"].to_numpy(),evaluation.select(column).to_numpy())
        coefficient=float(head.named_steps['logisticregression'].coef_[0,0])
        reconstructed=evaluation.select('pair_id').with_columns(pl.Series('probability',probability)).sort('pair_id')
        saved=pl.read_parquet(config.PROCESSED/'actr_diagnostics/predictions'/f'{model}_score_only_T2_test.parquet').sort('pair_id')
        assert np.array_equal(reconstructed['pair_id'].to_numpy(),saved['pair_id'].to_numpy())
        assert np.allclose(reconstructed['probability'],saved['probability'],rtol=1e-7,atol=1e-9),"Diagnostic head differs from saved evaluation"
        for split in ('train','validation','test'):
            frame=cohort.filter(pl.col('split')==split)
            raw=frame[column].to_numpy()
            auc=user_auc_table(frame['uid'],frame['label_c'],raw)
            rows.append({'model':model,'split':split,'lower_days':config.LOWER_BOUND_DAYS,
                'scaled_head_coefficient':coefficient,'head_direction':'positive' if coefficient>0 else 'negative' if coefficient<0 else 'constant',
                'raw_global_auc':float(roc_auc_score(frame['label_c'],raw)),'raw_user_auc':auc['auc'].mean()})
    pl.DataFrame(rows).write_csv(config.RESULTS/'actr_component_head_directions.csv')
    cohort.group_by('split','organic_context_available').agg(pl.len().alias('pairs'),
        pl.col('label_c').mean().alias('positive_rate'),pl.col('organic_sp').median().alias('median_organic_sp')
    ).write_csv(config.RESULTS/'actr_organic_context_groups.csv')


def _run_component_analysis():
    outputs=["actr_component_metrics.csv","actr_component_diagnostics.csv",
             "actr_component_paired_vs_audio.csv","actr_component_protocol.json"]
    if all((config.RESULTS/name).exists() for name in outputs):
        return pl.read_csv(config.RESULTS/outputs[0]),pl.read_csv(config.RESULTS/outputs[1])
    features=build_component_features()
    metrics=evaluate_features(features,DIAGNOSTIC_MODELS,"actr_component_metrics.csv",tasks=("T2",),
        artifact_root=config.PROCESSED/"actr_diagnostics")
    from .labels import with_labels
    pairs=with_labels(benchmark_pairs().filter(pl.col("task")=="T2").join(features,on="pair_id"))
    diagnostics=pairs.group_by("split").agg(pl.len().alias("pairs"),
        (pl.col("s_only")==0).mean().alias("s_target_exact_zero_fraction"),
        (pl.col("s_self_mass")>.99).mean().alias("s_self_mass_above_099_fraction"),
        (pl.col("s_max_mass")>.99).mean().alias("s_max_mass_above_099_fraction"),
        pl.col("context_is_organic").mean().alias("organic_context_fraction"),
        pl.col("partial_audio_available").mean().alias("partial_audio_available_fraction"),
        pl.col("organic_context_available").mean().alias("organic_context_available_fraction"),
        pl.col("candidate_count").median().alias("median_candidate_count"))
    diagnostics.write_csv(config.RESULTS/"actr_component_diagnostics.csv")
    from .metrics import paired_user_auc_difference
    paired=[]
    for lower in config.SENSITIVITY_LOWER_DAYS:
        for protocol in ("score_only","baseline_plus_score"):
            directory=config.PROCESSED/"actr_diagnostics/user_auc"
            reference=pl.read_parquet(directory/f"AudioCosine_{protocol}_T2_c_test_l{lower}.parquet")
            for model in DIAGNOSTIC_MODELS:
                if model=="AudioCosine":continue
                table=pl.read_parquet(directory/f"{model}_{protocol}_T2_c_test_l{lower}.parquet")
                paired.append({"model":model,"reference":"AudioCosine","protocol":protocol,"lower_days":lower,
                    "task":"T2","route":"C","split":"test",**paired_user_auc_difference(table,reference)})
    pl.DataFrame(paired).write_csv(config.RESULTS/"actr_component_paired_vs_audio.csv")
    (config.RESULTS/"actr_component_protocol.json").write_text(json.dumps({
        "purpose":"post-result exploratory diagnosis; main nine models and L unchanged","task":"T2",
        "S_only":"original all-route session lift followed by softmax",
        "P_only":"original audio cosine against last strictly pre-anchor item followed by softmax",
        "organic_SP":"fit sessions after filtering organic events; last organic context; organic-history candidates+target; no base-level in T2",
        "raw_diagnostics":"unnormalized lift/cosine and candidate count use identical shared head; diagnostic only",
        "missing_organic_context":"zero raw S/P; uniform softmax on candidate set",
        "embedding":{"file":"Yambda embeddings.parquet","column":"normalized_embed","dimensions":128,
                     "kind":"L2-normalized audio CNN embeddings; same field as AudioCosine","cache_dtype":"float32"},
        "model_days":[config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY],
        "lower_days":list(config.SENSITIVITY_LOWER_DAYS),"reference_reconstruction":"S+P asserted equal to saved original for every pair and L",
    },indent=2)+"\n")
    return metrics,diagnostics
