"""主結果の集約、A/B/Cの順位とRQ3。テスト結果で設定を選ばない。"""
import json
import re
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import softmax
from scipy.stats import kendalltau

from . import config
from .data import connection
from .labels import with_labels
from .light_models import (_audio_arrays,benchmark_pairs,fit_item_sessions,history_before,ndcg_for_ranking,top_ranked_indices)
from .metrics import bootstrap_mean,evaluate_probabilities,evaluate_ranking,paired_user_auc_difference,user_auc_table
from .official_models import FINAL_SEEDS,full_catalog_ndcg


def family_name(name):
    return re.sub(r"_seed\d+$","",name)


def collect_model_results():
    if not (config.RESULTS/'score_protocol_completed.json').exists():
        raise RuntimeError('Raw-score protocol must finish before aggregation')
    paths=list(config.RESULTS.glob("*_model_metrics.csv"))
    if not paths:raise RuntimeError("モデルの結果がない")
    raw=pl.concat([pl.read_csv(p) for p in paths],how="diagonal_relaxed").unique(
        subset=["model","protocol","task","route","split","lower_days"])
    raw=raw.with_columns(pl.col("model").map_elements(family_name,return_dtype=pl.String).alias("family"))
    raw=raw.filter(pl.col("family").is_in([*config.BENCHMARK_MODELS,"UserBaseline"]))
    keys=["family","protocol","task","route","split","lower_days","is_main_lower"]
    numeric=[c for c in raw.columns if raw.schema[c].is_numeric() and c not in keys]
    summary=raw.group_by(keys).agg([pl.col(c).mean().alias(c) for c in numeric]+[
        pl.col("model").n_unique().alias("seeds"),pl.col("user_auc").std(ddof=0).alias("seed_user_auc_std")])
    rows=[]
    for row in summary.iter_rows(named=True):
        if row["seeds"]>1:
            tables=[]
            for seed in FINAL_SEEDS:
                name=f"{row['family']}_seed{seed}_{row['protocol']}_{row['task']}_{row['route'].lower()}_{row['split']}_l{row['lower_days']}.parquet"
                directory=config.PROCESSED/('raw_scores' if row['protocol']=='raw_score' else '')/'user_auc'
                table=pl.read_parquet(directory/name).select("uid","auc")
                tables.append(table)
            averaged=pl.concat(tables).group_by("uid").agg(pl.col("auc").mean()).sort("uid")
            row["user_auc"]=averaged["auc"].mean()
            row["user_auc_ci_low"],row["user_auc_ci_high"]=bootstrap_mean(averaged["auc"])
        rows.append(row)
    table=pl.DataFrame(rows).sort("task","route","lower_days","split","protocol","family")
    assert {config.LOWER_BOUND_DAYS,7,14}.issubset(set(table["lower_days"]))
    table.write_csv(config.RESULTS/"all_model_results.csv")
    main=table.filter((pl.col("lower_days")==config.LOWER_BOUND_DAYS)&(pl.col("route")=="C")&(pl.col("split")=="test"))
    main.write_csv(config.RESULTS/"main_test_results.csv")
    _plot_lower_sensitivity(table)
    return table


def _plot_lower_sensitivity(table):
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    data=table.filter((pl.col("route")=="C")&(pl.col("split")=="test")&(pl.col("protocol")=="raw_score"))
    for ax,task in zip(axes,("T2","T1")):
        for name in sorted(data["family"].unique()):
            sub=data.filter((pl.col("task")==task)&(pl.col("family")==name)).sort("lower_days")
            ax.plot(sub["lower_days"],sub["user_auc"],marker="o",label=name)
        ax.axhline(.5,color="grey",linestyle="--",linewidth=.8)
        ax.set(title=task,xlabel="Lower bound L (days)",xticks=list(config.SENSITIVITY_LOWER_DAYS))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Raw-score within-user AUC")
    axes[1].legend(fontsize=7,bbox_to_anchor=(1.02,1),loc="upper left")
    fig.tight_layout()
    fig.savefig(config.RESULTS/"figures/model_lower_bound_sensitivity.png",dpi=170)
    plt.close(fig)


def paired_model_comparisons():
    """全モデル完成後、主Cの同一ペア上のユーザー別AUC差を測る。"""
    required=set(config.BENCHMARK_MODELS)
    sansa_status=config.RESULTS/"sansa_status.csv"
    if sansa_status.exists() and pl.read_csv(sansa_status)["status"].str.starts_with("not_run_").all():
        required.discard("SANSA")
    from .reporting import MODEL_FILES
    result_paths=[config.RESULTS/MODEL_FILES.get(m,f"{m.lower()}_model_metrics.csv") for m in sorted(required)]
    result_paths.append(config.RESULTS/'raw_score_model_metrics.csv')
    missing_results=[str(path) for path in result_paths if not path.exists()]
    if missing_results:raise RuntimeError("All-model paired comparison is waiting for: "+", ".join(missing_results))
    specifications=[]
    for task in ("T2","T1"):
        for protocol in ("raw_score","score_only","baseline_plus_score"):
            for model in sorted(required):
                names=[f"{model}_seed{seed}" for seed in FINAL_SEEDS] if model in ("iALS","BPR") else [model]
                directory=config.PROCESSED/('raw_scores' if protocol=='raw_score' else '')/'predictions'
                paths=[directory/f"{name}_{protocol}_{task}_test.parquet" for name in names]
                specifications.append((task,protocol,model,paths))
    missing=[str(p) for task,protocol,model,paths in specifications for p in paths if not p.exists()]
    if missing:raise RuntimeError("All-model paired comparison is waiting for: "+", ".join(missing))
    references={};averaged={}
    for task,protocol,model,paths in specifications:
        tables=[]
        for path in paths:
            frame=pl.read_parquet(path)
            identity=frame.select("pair_id","uid","label_c").sort("pair_id")
            if identity["pair_id"].n_unique()!=identity.height:raise ValueError(f"Duplicate pairs: {path}")
            if task in references:
                if not identity.equals(references[task]):raise ValueError(f"Paired cohorts differ: {path}")
            else:references[task]=identity
            auc=user_auc_table(frame["uid"],frame["label_c"],frame['score' if protocol=='raw_score' else 'probability'])
            tables.append(auc.select("uid","auc"))
        averaged[task,protocol,model]=pl.concat(tables).group_by("uid").agg(pl.col("auc").mean()).sort("uid")
    rows=[];top_rows=[]
    for task in ("T2","T1"):
        for protocol in ("raw_score","score_only","baseline_plus_score"):
            models=sorted(required,key=lambda m:(-averaged[task,protocol,m]["auc"].mean(),m))
            for left_index,right_index in combinations(range(len(models)),2):
                left,right=models[left_index],models[right_index]
                row={"task":task,"protocol":protocol,"route":"C","lower_days":config.LOWER_BOUND_DAYS,
                    "left_model":left,"right_model":right,"left_rank":left_index+1,"right_rank":right_index+1,
                    "left_user_auc":averaged[task,protocol,left]["auc"].mean(),
                    "right_user_auc":averaged[task,protocol,right]["auc"].mean(),
                    **paired_user_auc_difference(averaged[task,protocol,left],averaged[task,protocol,right])}
                rows.append(row)
                if right_index<3:top_rows.append(row)
    table=pl.DataFrame(rows)
    table.write_csv(config.RESULTS/"paired_model_differences.csv")
    pl.DataFrame(top_rows).write_csv(config.RESULTS/"top_model_paired_differences.csv")
    (config.RESULTS/"paired_model_protocol.json").write_text(json.dumps({
        "task":["T2","T1"],"route":"C","lower_days":config.LOWER_BOUND_DAYS,
        "primary_protocol":"raw_score; higher is better as designed, no sign flip",
        "secondary_protocols":["score_only","baseline_plus_score"],
        "models":sorted(required),"cohorts":"identical pair_id,uid,label_c across models and seeds asserted",
        "bootstrap_unit":"user; sample paired per-user AUC differences with replacement",
        "bootstrap_replicates":1000,"seed":config.SEED,"confidence_interval":"95% percentile; unadjusted",
        "multiple_seeds":"average AUC within each user across all three seeds before differences",
        "top_table":"top3 by test macro user AUC within each task/protocol; posthoc view of all pairs",
        "interpretation":"positive difference favors left; interval containing0 does not establish superiority",
    },indent=2)+"\n")
    return table


def model_cold_coverage():
    """raw全履歴の未出現率に加え、各モデルの学習フィルタ後の最低スコア率。"""
    rows=[]
    pairs=benchmark_pairs()
    for model in ("ItemKNN","iALS","BPR","SASRec"):
        path=config.PROCESSED/f"{model}_features.parquet"
        if not path.exists():continue
        feature=pl.read_parquet(path).select("pair_id",f"{model}_cold")
        frame=pairs.join(feature,on="pair_id")
        for lower in config.SENSITIVITY_LOWER_DAYS:
            labeled=with_labels(frame,lower)
            summary=labeled.group_by("task","split").agg(pl.len().alias("pairs"),
                pl.col(f"{model}_cold").sum().alias("cold_pairs"))
            for row in summary.iter_rows(named=True):
                rows.append({"model":model,"lower_days":lower,**row,"cold_fraction":row["cold_pairs"]/row["pairs"]})
    result=pl.DataFrame(rows)
    result.write_csv(config.RESULTS/"model_cold_coverage.csv")
    return result


def evaluate_task_a():
    """149日目モデルで150〜210日の全カタログholdout NDCG@10。"""
    events=history_before(config.REPRESENTATION_END_DAY).collect()
    eligible=events.filter(pl.col("played_ratio_pct")>=50).group_by("uid").len().filter(pl.col("len")>=2).select("uid")
    targets=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").filter(
        (pl.col("timestamp")>=150*config.SECONDS_PER_DAY)&(pl.col("timestamp")<211*config.SECONDS_PER_DAY)
        &(pl.col("played_ratio_pct")>=50)).select("uid","item_id").collect().join(eligible,on="uid",how="semi")
    directory=config.PROCESSED/f"models/day_{config.REPRESENTATION_END_DAY}"
    rows=[]
    pop=pl.read_parquet(directory/"popularity.parquet")
    for model,column in (("MostPop","mostpop"),("DecayPop","decaypop")):
        ranked=pop.sort([column,"item_id"],descending=[True,False])["item_id"].head(10).to_list()
        rows.append({"model":model,"seed":42,"ndcg_at_10":ndcg_for_ranking(ranked,targets,k=10)})
    if (directory/"itemknn_users.npz").exists():
        from scipy import sparse
        fit=np.load(directory/"itemknn_users.npz")
        score=full_catalog_ndcg(fit["vectors"],sparse.load_npz(directory/"itemknn_items.npz"),fit["users"],fit["items"],targets,k=10)
        rows.append({"model":"ItemKNN","seed":42,"ndcg_at_10":score})
    for model in ("iALS","BPR"):
        for seed in FINAL_SEEDS:
            path=directory/f"{model}_seed{seed}.npz"
            if path.exists():
                fit=np.load(path)
                value=full_catalog_ndcg(fit["user_vectors"],fit["item_vectors"],fit["users"],fit["items"],targets,k=10)
                rows.append({"model":model,"seed":seed,"ndcg_at_10":value})
    # 音響とACT-Rは同じ149日以前のユーザー表現を使い、全50Mカタログを採点する。
    rows.extend(_content_memory_task_a(events,targets))
    if (directory/"SASRec.pt").exists():
        rows.append({"model":"SASRec","seed":42,"ndcg_at_10":_sasrec_task_a(directory,targets)})
    table=pl.DataFrame(rows)
    table.write_csv(config.RESULTS/"task_a_next_item.csv")
    (config.RESULTS/"task_a_protocol.json").write_text(json.dumps({
        "fit_end_day":149,"target_days":[150,210],"target_filter":"played_ratio_pct>=50",
        "targets":"unique held-out items per user","users":"at least2 eligible pre150 listens",
        "candidate_ranking":"full catalog, no sampled negatives; CF unknown items get minimum score",
        "user_input":"frozen pre150 history; no post149 input","metric":"macro binary NDCG@10",
        "note":"Yambda-style multi-positive temporal holdout, not per-event online next-track accuracy",
    },indent=2)+"\n")
    return table


def _content_memory_task_a(events,targets):
    audio_items,audio=_audio_arrays()
    all_items=pl.read_parquet(config.PROCESSED/"items_50m.parquet")["item_id"].sort().to_numpy()
    ai=np.searchsorted(all_items,audio_items)
    vectors=np.zeros((len(all_items),audio.shape[1]),dtype=np.float32)
    vectors[ai]=audio
    trained_items,item_sessions=fit_item_sessions(config.REPRESENTATION_END_DAY)
    session_items=item_sessions.T.tocsr()
    prevalence=np.asarray(item_sessions.sum(axis=1)).ravel()
    trained_positions=np.searchsorted(all_items,trained_items)
    references={int(uid):set(values) for uid,values in targets.group_by("uid").agg(pl.col("item_id").unique()).iter_rows()}
    has_embedding=np.zeros(len(all_items),dtype=bool)
    has_embedding[ai]=True
    users=[];means=[];histories=[];has_profile=[]
    for h in events.sort("uid","timestamp","item_id").partition_by("uid",maintain_order=True):
        uid=int(h["uid"][0])
        if uid not in references:continue
        ids=np.searchsorted(all_items,h["item_id"].to_numpy())
        organic=h["is_organic"].to_numpy()==1
        mean=vectors[ids[organic]].sum(axis=0)
        has_profile.append(np.linalg.norm(mean)>0)
        mean/=max(np.linalg.norm(mean),1e-12)
        means.append(mean);users.append(uid);histories.append((h,ids))
    means=np.array(means,dtype=np.float32)
    audio_values=[];actr_values=[]
    reference=(config.REPRESENTATION_END_DAY+1+config.LOWER_BOUND_DAYS)*config.SECONDS_PER_DAY
    for start in range(0,len(users),32):
        # 32ユーザーずつで音響の全曲スコアを計算し、密行列を保持しない。
        cosine=means[start:start+32]@vectors.T
        cosine[:,~has_embedding]=-1
        for j,available in enumerate(has_profile[start:start+32]):
            if not available:cosine[j,:]=-1
        context_vectors=np.array([vectors[ids[-1]] for h,ids in histories[start:start+32]])
        partial=context_vectors@vectors.T
        for j,uid in enumerate(users[start:start+32]):
            h,ids=histories[start+j]
            relevant=references[uid]
            audio_values.append(_ndcg_one(cosine[j],all_items,relevant))
            ages=(reference-h["timestamp"].to_numpy())/config.SECONDS_PER_DAY
            base=np.bincount(ids,weights=ages**-.5,minlength=len(all_items))
            context=h["item_id"][-1]
            context_index=np.searchsorted(trained_items,context)
            spreading=np.zeros(len(all_items),dtype=np.float32)
            if context_index<len(trained_items) and trained_items[context_index]==context:
                sessions=item_sessions[context_index].indices
                co=np.asarray(session_items[sessions].sum(axis=0)).ravel()
                spreading[trained_positions]=co*item_sessions.shape[1]/(prevalence*prevalence[context_index])
            activation=softmax(base)+softmax(spreading)+softmax(partial[j])
            actr_values.append(_ndcg_one(activation,all_items,relevant))
        if start%320==0:print(f"task A audio/ACT-R {start}/{len(users)} users",flush=True)
    return [{"model":"AudioCosine","seed":42,"ndcg_at_10":float(np.mean(audio_values))},
            {"model":"ACT-R","seed":42,"ndcg_at_10":float(np.mean(actr_values))}]


def _ndcg_one(scores,items,relevant):
    top=top_ranked_indices(scores,items,10)
    dcg=sum(1/np.log2(rank+2) for rank,index in enumerate(top) if items[index] in relevant)
    ideal=(1/np.log2(np.arange(2,min(10,len(relevant))+2))).sum()
    return float(dcg/ideal)


def _sasrec_task_a(directory,targets):
    import torch
    from .sequence_model import PARAMETERS,_batch_inputs,device_name
    from .vendor.sasrec_model import SASRecEncoder
    items=np.load(directory/"SASRec_items.npy")
    model=SASRecEncoder(num_items=len(items),**PARAMETERS).to(device_name())
    model.load_state_dict(torch.load(directory/"SASRec.pt",map_location=device_name(),weights_only=True))
    model._encoder.use_nested_tensor=False
    model.eval()
    sequences=pl.read_parquet(directory/"SASRec_train_sequences.parquet")
    users=sequences["uid"].to_numpy()
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(users),128):
            batch=sequences["item_id"].slice(start,128).to_list()
            vectors.append(model(_batch_inputs(batch,device_name())).cpu().numpy())
    uv=np.concatenate(vectors)
    iv=model.item_embeddings.weight[1:].detach().cpu().numpy()
    return full_catalog_ndcg(uv,iv,users,items,targets,k=10)


def rank_agreements(results,task_a):
    a=task_a.group_by("model").agg(pl.col("ndcg_at_10").mean()).rename({"model":"family"})
    joined=results.filter((pl.col("task")=="T2")&(pl.col("protocol")=="raw_score")
        &(pl.col("split")=="test")&(pl.col("lower_days")==config.LOWER_BOUND_DAYS)).select(
            "family","route","user_auc").pivot(on="route",index="family",values="user_auc").join(a,on="family")
    rows=[]
    for left,right in (("ndcg_at_10","C"),("B","C")):
        tau=kendalltau(joined[left],joined[right])
        rows.append({"comparison":f"{left}_vs_{right}","models":len(joined),"kendall_tau":float(tau.statistic),"p_value":float(tau.pvalue)})
    joined.write_csv(config.RESULTS/"model_rank_comparison.csv")
    table=pl.DataFrame(rows)
    table.write_csv(config.RESULTS/"kendall_rank_agreement.csv")
    return table


def rq3_strata():
    pairs=with_labels(benchmark_pairs()).filter(pl.col("split")=="test")
    features=pl.read_parquet(config.PROCESSED/"light_features.parquet").select("pair_id","mostpop","audio_cosine","audio_missing")
    pairs=pairs.join(features,on="pair_id")
    # 履歴長は起点より前の再生数+今回の1件。likesは記述的な事後反応のみ。
    history=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").select("uid","timestamp").sort("uid","timestamp").collect()
    lengths=[]
    groups={h["uid"][0]:h["timestamp"].to_numpy() for h in history.partition_by("uid",maintain_order=True)}
    for g in pairs.partition_by("uid",maintain_order=True):
        values=np.searchsorted(groups[g["uid"][0]],g["anchor_timestamp"].to_numpy(),side="left")+1
        lengths.append(g.select("pair_id").with_columns(pl.Series("history_length",values)))
    pairs=pairs.join(pl.concat(lengths),on="pair_id")
    with connection() as con:
        con.register("pairs",pairs.select("pair_id","uid","item_id","anchor_timestamp").to_arrow())
        con.from_parquet(str(config.RAW/"flat/50m/likes.parquet")).create_view("likes")
        likes=con.execute(f"""SELECT p.pair_id,count(l.timestamp)>0 AS liked_within_24h
            FROM pairs p LEFT JOIN likes l ON p.uid=l.uid AND p.item_id=l.item_id
            AND l.timestamp>=p.anchor_timestamp AND l.timestamp<=p.anchor_timestamp+{config.SECONDS_PER_DAY}
            GROUP BY p.pair_id""").pl()
    pairs=pairs.join(likes,on="pair_id")
    columns={"activity":f"active_days_l{config.LOWER_BOUND_DAYS}_h{config.HORIZON_DAYS}",
             "popularity":"mostpop","history":"history_length","closeness":"audio_cosine"}
    definitions={}
    for name,column in columns.items():
        values=pairs[column].to_numpy()
        available=~pairs["audio_missing"].to_numpy() if name=="closeness" else (values>0 if name=="popularity" else np.ones(len(values),dtype=bool))
        cuts=np.unique(np.quantile(values[available],[.25,.5,.75]))
        definitions[name]={"column":column,"cuts":cuts.tolist(),"cohort":"test features only; no label-based boundary"}
        groups=[f"Q{x+1}" if known else "missing_or_unseen" for x,known in zip(np.searchsorted(cuts,values,side="left"),available)]
        pairs=pairs.with_columns(pl.Series(f"stratum_{name}",groups))
    pairs=pairs.with_columns(
        pl.when(pl.col("played_ratio_pct")<50).then(pl.lit("<50%"))
        .when(pl.col("played_ratio_pct")<100).then(pl.lit("50–99%"))
        .when(pl.col("played_ratio_pct")==100).then(pl.lit("100%"))
        .otherwise(pl.lit(">100%")).alias("stratum_played_ratio"),
        pl.col("liked_within_24h").cast(pl.String).alias("stratum_like"))
    rows=[];user_tables={}
    paths=[*sorted((config.PROCESSED/'raw_scores/predictions').glob('*_test.parquet')),
        *sorted((config.PROCESSED/"predictions").glob("*_test.parquet"))]
    for path in paths:
        data=pl.read_parquet(path)
        if 'pair_id' not in data.columns:continue
        name=path.name.removesuffix("_test.parquet")
        match=re.fullmatch(r"(.+)_(raw_score|score_only|baseline_plus_score|baseline)_(T[12])",name)
        if match is None:raise ValueError(f"Unknown prediction name: {name}")
        model,protocol,task=match.groups()
        if family_name(model) not in config.BENCHMARK_MODELS and model!="UserBaseline":continue
        score_column='score' if protocol=='raw_score' else 'probability'
        sub=pairs.join(data.select('pair_id',score_column),on='pair_id')
        for stratifier in list(columns)+["played_ratio","like"]:
            column=f"stratum_{stratifier}"
            for group in sub[column].unique():
                g=sub.filter(pl.col(column)==group)
                evaluator=evaluate_ranking if protocol=='raw_score' else evaluate_probabilities
                metric,auc=evaluator(g['uid'].to_numpy(),g['label_c'].to_numpy(),g[score_column].to_numpy(),n_bootstrap=1000)
                keys={"family":family_name(model),"protocol":protocol,"task":task,"stratifier":stratifier,
                      "stratum":group,"lower_days":config.LOWER_BOUND_DAYS}
                rows.append({"prediction":name,**keys,**metric})
                user_tables.setdefault(tuple(keys.values()),[]).append(auc.select("uid","auc"))
    raw=pl.from_dicts(rows,infer_schema_length=None)
    raw.write_csv(config.RESULTS/"rq3_metrics.csv")
    keys=["family","protocol","task","stratifier","stratum","lower_days"]
    numeric=[c for c in raw.columns if raw.schema[c].is_numeric() and c not in keys]
    summary=raw.group_by(keys).agg([pl.col(c).mean() for c in numeric]+[pl.len().alias("seeds")])
    averaged=[]
    for row in summary.iter_rows(named=True):
        auc=pl.concat(user_tables[tuple(row[k] for k in keys)]).group_by("uid").agg(pl.col("auc").mean()).sort("uid")
        row["user_auc"]=auc["auc"].mean()
        row["user_auc_ci_low"],row["user_auc_ci_high"]=bootstrap_mean(auc["auc"])
        averaged.append(row)
    pl.DataFrame(averaged).write_csv(config.RESULTS/"rq3_summary.csv")
    definitions["like"]={"window":"[anchor,anchor+24h]","descriptive_only":True,"used_as_predictor":False}
    definitions["activity"]["descriptive_only"]=True
    (config.RESULTS/"rq3_strata.json").write_text(json.dumps(definitions,indent=2)+"\n")
    return pl.DataFrame(averaged)
