"""期限を分けた人気度と、起点までの音響・記憶スコア。"""
import json

import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from . import config
from .data import connection, prepare_audio_embeddings


DECAY_HOURS = (.5, 1, 2, 3, 6, 12, 24)


def top_ranked_indices(scores,items,k):
    """全曲を採点し、同点はitem_idの小さい順にする。"""
    k=min(k,len(scores))
    if k<=0:return np.empty(0,dtype=np.int64)
    threshold=np.partition(scores,-k)[-k]
    above=np.flatnonzero(scores>threshold)
    tied=np.flatnonzero(scores==threshold)
    needed=k-len(above)
    tie_order=np.argpartition(items[tied],needed-1)[:needed]
    chosen=np.r_[above,tied[tie_order]]
    return chosen[np.lexsort((items[chosen],-scores[chosen]))]


def benchmark_pairs():
    return pl.read_parquet(config.PROCESSED / "pairs.parquet").filter(
        pl.col("task").is_in(["T1", "T2"])).with_row_index("pair_id")


def history_before(end_day):
    return pl.scan_parquet(config.RAW / "flat/50m/listens.parquet").filter(
        pl.col("timestamp").cast(pl.Int64)<(end_day+1)*config.SECONDS_PER_DAY)


def popularity_table(events, hour):
    if not events.height:
        raise ValueError("人気度を学習する履歴が空")
    if not hour:
        return events.group_by("item_id").len(name="score").with_columns(pl.col("score").cast(pl.Float64))
    latest = events["timestamp"].max()
    return events.with_columns(
        (.9**((latest-pl.col("timestamp").cast(pl.Int64))/(hour*3600))).alias("weight")
    ).group_by("item_id").agg(pl.col("weight").sum().alias("score"))


def ndcg_for_ranking(ranked_items, validation, k=100):
    """公式と同じ全学習カタログ順位・ユーザー別の複数正解で調整する。"""
    ranks = {item: rank for rank,item in enumerate(ranked_items[:k],1)}
    values=[]
    for targets in validation.group_by("uid").agg(pl.col("item_id").unique())["item_id"]:
        if not len(targets):
            continue
        dcg=sum(1/np.log2(ranks[item]+1) for item in targets if item in ranks)
        ideal=(1/np.log2(np.arange(2,min(k,len(targets))+2))).sum()
        values.append(dcg/ideal)
    return float(np.mean(values)) if values else float("nan")


def fit_popularity(end_day):
    directory=config.PROCESSED/f"models/day_{end_day}"
    directory.mkdir(parents=True,exist_ok=True)
    # 最後の1日を調整用にし、30分の空白を置く。すべて期限内。
    cutoff=(end_day+1)*config.SECONDS_PER_DAY
    events=history_before(end_day).filter(pl.col("played_ratio_pct")>=50).select("uid","item_id","timestamp").collect()
    training=events.filter(pl.col("timestamp")<cutoff-config.SECONDS_PER_DAY-1800)
    valid=events.filter(pl.col("timestamp")>=cutoff-config.SECONDS_PER_DAY).join(
        training.select("uid").unique(),on="uid",how="semi")
    trials=[]
    for hour in DECAY_HOURS:
        scores=popularity_table(training,hour).sort(["score","item_id"],descending=[True,False])
        trials.append({"hour":hour,"ndcg_at_100":ndcg_for_ranking(scores["item_id"].head(100).to_list(),valid)})
    best=max(trials,key=lambda row:row["ndcg_at_100"])["hour"]
    pl.DataFrame(trials).write_csv(config.RESULTS/f"decaypop_tuning_day_{end_day}.csv")
    table=popularity_table(events,0).rename({"score":"mostpop"}).join(
        popularity_table(events,best).rename({"score":"decaypop"}),on="item_id")
    table.write_parquet(directory/"popularity.parquet")
    metadata={"fit_end_day":end_day,"fit_max_timestamp":events["timestamp"].max(),"best_hour":best,
              "tuning_train_exclusive_end":cutoff-config.SECONDS_PER_DAY-1800,
              "tuning_validation_start":cutoff-config.SECONDS_PER_DAY,"tuning_metric":"full_catalog_ndcg@100",
              "played_ratio_threshold":50,"trials":len(DECAY_HOURS)}
    (directory/"popularity.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return table


def baseline_user_table(end_day):
    """期限までに確定した1〜7日再聴率と、直近30日の活動日数。"""
    cutoff=(end_day+1)*config.SECONDS_PER_DAY
    start=max(0,cutoff-30*config.SECONDS_PER_DAY)
    with connection() as con:
        con.from_parquet(str(config.RAW/"flat/50m/listens.parquet")).create_view("events")
        con.execute(f"CREATE TEMP TABLE past AS SELECT * FROM events WHERE timestamp<{cutoff}")
        table=con.execute(f"""WITH anchors AS (
            SELECT uid,item_id,min(timestamp)::BIGINT AS t FROM past WHERE is_organic=0
            GROUP BY uid,item_id HAVING t>={start} AND t+{7*config.SECONDS_PER_DAY}<{cutoff}
        ), returns AS (
            SELECT a.uid,a.item_id,a.t,count(e.timestamp)>0 AS returned FROM anchors a
            LEFT JOIN past e ON e.uid=a.uid AND e.item_id=a.item_id AND e.is_organic=1
                AND e.timestamp>=a.t+{config.SECONDS_PER_DAY} AND e.timestamp<=a.t+{7*config.SECONDS_PER_DAY}
            GROUP BY a.uid,a.item_id,a.t
        ), rates AS (SELECT uid,count(*) AS exposures,sum(returned::INTEGER) AS returns FROM returns GROUP BY uid),
        activity AS (SELECT uid,count(DISTINCT timestamp//{config.SECONDS_PER_DAY}) AS active_days
            FROM past WHERE timestamp>={start} GROUP BY uid)
        SELECT activity.uid,activity.active_days,coalesce(rates.exposures,0) AS exposures,
            coalesce(rates.returns,0) AS returns FROM activity LEFT JOIN rates USING(uid)
        """).pl()
    prior=float(table["returns"].sum()/max(1,table["exposures"].sum()))
    table=table.with_columns(((pl.col("returns")+20*prior)/(pl.col("exposures")+20)).alias("past_return_rate"))
    table.write_parquet(config.PROCESSED/f"models/day_{end_day}/user_baseline.parquet")
    return table.select("uid","past_return_rate",pl.col("active_days").alias("past_active_days"))


def fit_item_sessions(end_day,organic_only=False):
    """ACT-R原典の30分セッション。曲×曲の密行列を作らない。"""
    events=history_before(end_day)
    if organic_only:events=events.filter(pl.col("is_organic")==1)
    events=events.select("uid","item_id",pl.col("timestamp").cast(pl.Int64)).sort("uid","timestamp")
    sessions=events.with_columns(
        (pl.col("timestamp").diff().over("uid").fill_null(1801)>1800).cast(pl.UInt32).alias("new_session")
    ).with_columns(pl.col("new_session").cum_sum().over("uid").alias("session_number")).select(
        (pl.col("uid").cast(pl.UInt64)*2**32+pl.col("session_number")).alias("session_key"),"item_id"
    ).unique().collect()
    items=np.sort(sessions["item_id"].unique().to_numpy())
    keys=np.sort(sessions["session_key"].unique().to_numpy())
    row=np.searchsorted(items,sessions["item_id"].to_numpy())
    col=np.searchsorted(keys,sessions["session_key"].to_numpy())
    matrix=sparse.csr_matrix((np.ones(len(row),dtype=np.float32),(row,col)),shape=(len(items),len(keys)))
    return items,matrix


def _audio_arrays():
    frame=pl.read_parquet(prepare_audio_embeddings()).sort("item_id")
    items=frame["item_id"].to_numpy()
    embeds=frame["normalized_embed"].to_numpy()
    matrix=np.stack(embeds).astype(np.float32)
    norms=np.linalg.norm(matrix,axis=1)
    assert np.isfinite(matrix).all() and np.max(np.abs(norms-1))<1e-4
    return items,matrix


def build_light_features():
    pairs=benchmark_pairs()
    output=pairs.select("pair_id","uid","item_id","anchor_timestamp","task","split")
    features=[]
    for day in (config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY):
        pop=fit_popularity(day)
        users=baseline_user_table(day)
        chosen=pairs.filter(pl.col("split")=="train" if day==config.HEAD_REPRESENTATION_END_DAY else pl.col("split")!="train")
        assert chosen["anchor_timestamp"].min()>=(day+1)*config.SECONDS_PER_DAY
        features.append(chosen.select("pair_id","uid","item_id").join(pop,on="item_id",how="left").join(users,on="uid",how="left").select(
            "pair_id",pl.lit(day).alias("model_end_day"),pl.col("mostpop").fill_null(0),pl.col("decaypop").fill_null(0),
            pl.col("past_return_rate").fill_null(0),pl.col("past_active_days").fill_null(0)))
    output=output.join(pl.concat(features),on="pair_id").sort("pair_id")
    audio_items,audio=_audio_arrays()
    # 全イベントをpandasに載せず、必要列のみ。ユーザーごとの小行列を順番に処理。
    history=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").select(
        "uid","item_id",pl.col("timestamp").cast(pl.Int64),"is_organic").filter(
        pl.col("timestamp")<=pairs["anchor_timestamp"].max()).collect().sort("uid","timestamp","item_id")
    result=[]
    for day in (config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY):
        sub=output.filter(pl.col("model_end_day")==day)
        trained_items,item_sessions=fit_item_sessions(day)
        prevalence=np.asarray(item_sessions.sum(axis=1)).ravel()
        n_sessions=item_sessions.shape[1]
        histories={group["uid"][0]:group for group in history.partition_by("uid",maintain_order=True)}
        groups=sub.partition_by("uid",maintain_order=True)
        with threadpool_limits(limits=4):
            for number,group in enumerate(groups):
                user=group["uid"][0]
                h=histories[user]
                result.append(_user_audio_actr(group,h,audio_items,audio,trained_items,item_sessions,prevalence,n_sessions))
                if number%500==0:
                    print(f"day {day}: audio/ACT-R {number}/{len(groups)} users",flush=True)
        del item_sessions,histories
    output=output.join(pl.concat(result),on="pair_id").sort("pair_id")
    output.write_parquet(config.PROCESSED/"light_features.parquet",compression="zstd")
    return output


def _user_audio_actr(pairs,history,audio_items,audio,trained_items,item_sessions,prevalence,n_sessions):
    times=history["timestamp"].to_numpy()
    item=history["item_id"].to_numpy()
    organic=history["is_organic"].to_numpy()==1
    anchors=pairs["anchor_timestamp"].to_numpy()
    targets=pairs["item_id"].to_numpy()
    tasks=pairs["task"].to_numpy()
    ends=np.searchsorted(times,anchors,side="left")
    local=np.unique(np.concatenate([item,targets]))
    indices=np.searchsorted(local,item)
    target_indices=np.searchsorted(local,targets)
    context=np.where(ends>0,item[np.maximum(ends-1,0)],0)
    contexts=np.unique(context)
    context_index=np.searchsorted(contexts,context)
    # 音響と共聴は固定したモデルのlookup。分母の候補集合は各起点の履歴+対象曲だけ。
    ai=np.searchsorted(audio_items,local)
    amask=(ai<len(audio_items))&(audio_items[np.minimum(ai,len(audio_items)-1)]==local)
    vectors=np.zeros((len(local),audio.shape[1]),dtype=np.float32)
    vectors[amask]=audio[ai[amask]]
    ci=np.searchsorted(local,contexts)
    context_vectors=np.zeros((len(contexts),audio.shape[1]),dtype=np.float32)
    cmask=(ci<len(local))&(local[np.minimum(ci,len(local)-1)]==contexts)
    context_vectors[cmask]=vectors[ci[cmask]]
    partial=vectors@context_vectors.T
    ti=np.searchsorted(trained_items,local)
    tm=(ti<len(trained_items))&(trained_items[np.minimum(ti,len(trained_items)-1)]==local)
    tc=np.searchsorted(trained_items,contexts)
    tcm=(tc<len(trained_items))&(trained_items[np.minimum(tc,len(trained_items)-1)]==contexts)
    spreading=np.zeros((len(local),len(contexts)),dtype=np.float32)
    if tm.any() and tcm.any():
        co=(item_sessions[ti[tm]]@item_sessions[tc[tcm]].T).toarray()
        lift=co*n_sessions/(prevalence[ti[tm],None]*prevalence[tc[tcm]][None,:])
        spreading[np.ix_(tm,tcm)]=lift
    rows={"pair_id":pairs["pair_id"].to_numpy(),"audio_cosine":[],"audio_missing":[]}
    lower_values=sorted(set(config.SENSITIVITY_LOWER_DAYS)|{config.LOWER_BOUND_DAYS})
    for lower in lower_values:
        rows[f"actr_l{lower}"]=[]
    # organic平均はイベントごと。未来のorganicイベントはprefixに入れない。
    organic_times=times[organic]
    organic_indices=indices[organic]
    organic_vectors=vectors[organic_indices]
    prefix=np.vstack([np.zeros((1,audio.shape[1]),dtype=np.float32),np.cumsum(organic_vectors,axis=0)])
    for j,(timestamp,end,target) in enumerate(zip(anchors,ends,target_indices)):
        organic_end=np.searchsorted(organic_times,timestamp,side="left")
        mean=prefix[organic_end]
        norm=np.linalg.norm(mean)
        missing=not amask[target] or norm==0
        rows["audio_cosine"].append(-1.0 if missing else float(vectors[target]@mean/norm))
        rows["audio_missing"].append(bool(missing))
        previous=indices[:end]
        candidates=np.unique(np.append(previous,target))
        target_position=int(np.searchsorted(candidates,target))
        col=context_index[j]
        sp=softmax(spreading[candidates,col])[target_position]+softmax(partial[candidates,col])[target_position]
        for lower in lower_values:
            if tasks[j]=="T2":
                activation=sp
            else:
                ages=(timestamp+lower*config.SECONDS_PER_DAY-times[:end])/config.SECONDS_PER_DAY
                base=np.bincount(previous,weights=ages**-.5,minlength=len(local))
                base[target]+=lower**-.5
                activation=sp+softmax(base[candidates])[target_position]
            rows[f"actr_l{lower}"].append(float(activation))
    return pl.DataFrame(rows)
