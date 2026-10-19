"""公式の行列モデルをCPUで、同じ探索範囲と二つの期限で学習する。"""
import json
import time

import numpy as np
import polars as pl
from scipy import sparse
from threadpoolctl import threadpool_limits

from . import config
from .light_models import benchmark_pairs, history_before, top_ranked_indices


KNN_HOURS=(0,.001,.002,.004,.008,.016,.032,.064,.128,.256,.5,1,2)
FINAL_SEEDS=(41,42,43)


def _suggest_parameters(trial,model):
    parameters={"iterations":trial.suggest_int("iterations",10,300,log=True),
                "regularization":trial.suggest_float("regularization",1e-5,1,log=True)}
    if model=="BPR":parameters["learning_rate"]=trial.suggest_float("learning_rate",1e-5,.1,log=True)
    return parameters


def _replayed_sampler(study,model):
    import optuna
    # 全10試行はTPEのstartup期間内。SQLiteの完了試行だけで乱数を確実に復元する。
    sampler=optuna.samplers.TPESampler(seed=42)
    replay=optuna.create_study(direction="maximize",sampler=sampler)
    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if len(completed)>10:raise ValueError("Expected at most10 completed trials")
    for previous in completed:
        trial=replay.ask()
        if _suggest_parameters(trial,model)!=previous.params:
            raise ValueError("Saved parameters differ from the fixed seed42 search; refusing unsafe resume")
        replay.tell(trial,previous.value)
    return sampler


def interaction_matrix(events,binary=False,users=None,items=None):
    if users is None:users=np.sort(events["uid"].unique().to_numpy())
    if items is None:items=np.sort(events["item_id"].unique().to_numpy())
    if binary:events=events.unique(subset=["uid","item_id"])
    row=np.searchsorted(users,events["uid"].to_numpy())
    col=np.searchsorted(items,events["item_id"].to_numpy())
    matrix=sparse.csr_matrix((np.ones(len(row),dtype=np.float32),(row,col)),shape=(len(users),len(items)))
    return users,items,matrix


def full_catalog_ndcg(user_vectors,item_vectors,users,items,validation,k=100):
    """サンプル候補を使わず全曲を採点する。未学習ユーザーは公式同様対象外。"""
    targets={int(row[0]):set(row[1]) for row in validation.group_by("uid").agg(pl.col("item_id").unique()).iter_rows()}
    results=[]
    with threadpool_limits(limits=4):
        for start in range(0,len(users),64):
            batch=user_vectors[start:start+64]
            scores=(item_vectors@batch.T).T if sparse.issparse(item_vectors) else batch@item_vectors.T
            for j,uid in enumerate(users[start:start+64]):
                relevant=targets.get(int(uid))
                if not relevant:continue
                top=top_ranked_indices(scores[j],items,k)
                ranked=items[top]
                hit=np.array([item in relevant for item in ranked],dtype=float)
                dcg=(hit/np.log2(np.arange(2,len(hit)+2))).sum()
                ideal=(1/np.log2(np.arange(2,min(k,len(relevant))+2))).sum()
                results.append((int(uid),float(dcg/ideal)))
    return float(np.mean([value for uid,value in results])) if results else float("nan")


def _split_for_tuning(end_day):
    cutoff=(end_day+1)*config.SECONDS_PER_DAY
    events=history_before(end_day).filter(pl.col("played_ratio_pct")>=50).select("uid","item_id","timestamp").collect()
    train=events.filter(pl.col("timestamp")<cutoff-config.SECONDS_PER_DAY-1800)
    valid=events.filter(pl.col("timestamp")>=cutoff-config.SECONDS_PER_DAY)
    return events,train,valid


def _knn_vectors(events,hour):
    users,items,matrix=interaction_matrix(events)
    last=events.group_by("uid").agg(pl.col("timestamp").max().alias("last"))
    weighted=events.join(last,on="uid").with_columns((pl.col("last").cast(pl.Int64)-pl.col("timestamp")).alias("delta"))
    delta=weighted["delta"].to_numpy()
    weights=(delta==0).astype(np.float32) if hour==0 else (.9**(delta/(hour*3600))).astype(np.float32)
    rows=np.searchsorted(users,weighted["uid"].to_numpy())
    cols=np.searchsorted(items,weighted["item_id"].to_numpy())
    recency=sparse.csr_matrix((weights,(rows,cols)),shape=matrix.shape)
    recency.data[recency.data<=1e-9]=0
    recency.eliminate_zeros()
    # 公式sparse_normalize(U.T, dim=-1)はユーザー列ごとに正規化する。
    norms=np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()+1e-12)
    item_vectors=(matrix.T@sparse.diags(1/norms)).tocsr()
    user_vectors=(recency@matrix.T).toarray()
    user_vectors/=np.maximum(np.linalg.norm(user_vectors,axis=1,keepdims=True),1e-12)
    return users,items,user_vectors,item_vectors


def fit_itemknn(end_day):
    directory=config.PROCESSED/f"models/day_{end_day}"
    directory.mkdir(parents=True,exist_ok=True)
    final,train,valid=_split_for_tuning(end_day)
    trials=[]
    for hour in KNN_HOURS:
        users,items,uv,iv=_knn_vectors(train,hour)
        metric=full_catalog_ndcg(uv,iv,users,items,valid)
        trials.append({"hour":hour,"ndcg_at_100":metric})
        print(f"ItemKNN day{end_day} h={hour} NDCG@100={metric:.5f}",flush=True)
    best=max(trials,key=lambda x:x["ndcg_at_100"])["hour"]
    pl.DataFrame(trials).write_csv(config.RESULTS/f"itemknn_tuning_day_{end_day}.csv")
    users,items,uv,iv=_knn_vectors(final,best)
    np.savez_compressed(directory/"itemknn_users.npz",users=users,items=items,vectors=uv)
    sparse.save_npz(directory/"itemknn_items.npz",iv)
    (directory/"itemknn.json").write_text(json.dumps({"end_day":end_day,"hour":best,"search_trials":len(KNN_HOURS),
        "max_timestamp":final["timestamp"].max(),"official_normalization":"U.T normalized per user column"},indent=2)+"\n")
    return users,items,uv,iv


def _matrix_fit(model,matrix,parameters,seed):
    if model=="iALS":
        from implicit.cpu.als import AlternatingLeastSquares
        learner=AlternatingLeastSquares(factors=64,regularization=parameters["regularization"],
            iterations=parameters["iterations"],random_state=seed,num_threads=4,calculate_training_loss=False)
    else:
        from implicit.cpu.bpr import BayesianPersonalizedRanking
        learner=BayesianPersonalizedRanking(factors=64,regularization=parameters["regularization"],
            iterations=parameters["iterations"],learning_rate=parameters["learning_rate"],random_state=seed,
            num_threads=4,verify_negative_samples=True)
    with threadpool_limits(limits=1,user_api="blas"):learner.fit(matrix,show_progress=False)
    return np.asarray(learner.user_factors),np.asarray(learner.item_factors)


def fit_matrix_factorization(model,end_day):
    import optuna
    directory=config.PROCESSED/f"models/day_{end_day}"
    directory.mkdir(parents=True,exist_ok=True)
    metadata_path=directory/f"{model}.json"
    if metadata_path.exists() and all((directory/f"{model}_seed{seed}.npz").exists() for seed in FINAL_SEEDS):
        return json.loads(metadata_path.read_text())
    final,train,valid=_split_for_tuning(end_day)
    users,items,matrix=interaction_matrix(train,binary=True)
    def objective(trial):
        parameters=_suggest_parameters(trial,model)
        begin=time.monotonic()
        uv,iv=_matrix_fit(model,matrix,parameters,42)
        result=full_catalog_ndcg(uv,iv,users,items,valid)
        print(f"{model} day{end_day} trial{trial.number}: {time.monotonic()-begin:.1f}s NDCG@100={result:.5f}",flush=True)
        return result
    # 公式の10試行を保持。公式で未固定の探索乱数は再現のため42に固定。
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42),study_name=model,
        storage=f"sqlite:///{directory / (model+'_study.sqlite3')}",load_if_exists=True)
    for trial in study.trials:
        if trial.state==optuna.trial.TrialState.RUNNING:
            study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
    study.sampler=_replayed_sampler(study,model)
    def checkpoint(study,trial):
        pl.DataFrame([{"trial":t.number,"value":t.value,"state":t.state.name,**t.params}
                      for t in study.trials]).write_csv(config.RESULTS/f"{model}_tuning_day_{end_day}.csv")
    completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,10-completed),callbacks=[checkpoint])
    users,items,matrix=interaction_matrix(final,binary=True)
    for seed in FINAL_SEEDS:
        path=directory/f"{model}_seed{seed}.npz"
        if path.exists():continue
        uv,iv=_matrix_fit(model,matrix,study.best_params,seed)
        temporary=path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,users=users,items=items,user_vectors=uv,item_vectors=iv)
        temporary.replace(path)
    metadata={"fit_end_day":end_day,"max_timestamp":final["timestamp"].max(),"parameters":study.best_params,
              "factors":64,"trials":10,"seeds":FINAL_SEEDS,"device":"CPU","search_seed":42}
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def _pair_scores(pairs,users,items,uv,iv):
    u=np.searchsorted(users,pairs["uid"].to_numpy())
    i=np.searchsorted(items,pairs["item_id"].to_numpy())
    known=(u<len(users))&(i<len(items))
    known &= (users[np.minimum(u,len(users)-1)]==pairs["uid"].to_numpy())
    known &= (items[np.minimum(i,len(items)-1)]==pairs["item_id"].to_numpy())
    if sparse.issparse(iv):floor=-1e-6
    else:floor=-float(np.linalg.norm(uv,axis=1).max()*np.linalg.norm(iv,axis=1).max())-1e-6
    scores=np.full(len(pairs),floor,dtype=np.float64)
    if sparse.issparse(iv):
        positions=np.flatnonzero(known)
        positions=positions[np.argsort(u[positions],kind="stable")]
        boundaries=np.r_[0,np.flatnonzero(np.diff(u[positions]))+1,len(positions)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            group=positions[lo:hi]
            if not len(group):continue
            user_index=u[group[0]]
            scores[group]=iv[i[group]]@uv[user_index]
    else:
        positions=np.flatnonzero(known)
        for start in range(0,len(positions),50000):
            batch=positions[start:start+50000]
            scores[batch]=np.einsum("ij,ij->i",uv[u[batch]],iv[i[batch]])
    return scores,~known


def build_official_features(model):
    pairs=benchmark_pairs()
    rows=[]
    for day in (config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY):
        chosen=pairs.filter(pl.col("split")=="train" if day==config.HEAD_REPRESENTATION_END_DAY else pl.col("split")!="train")
        assert chosen["anchor_timestamp"].min()>=(day+1)*config.SECONDS_PER_DAY
        directory=config.PROCESSED/f"models/day_{day}"
        columns={"pair_id":chosen["pair_id"].to_numpy()}
        if model=="ItemKNN":
            users,items,uv,iv=fit_itemknn(day)
            score,missing=_pair_scores(chosen,users,items,uv,iv)
            columns["itemknn"]=score
        else:
            fit_matrix_factorization(model,day)
            for seed in FINAL_SEEDS:
                fitted=np.load(directory/f"{model}_seed{seed}.npz")
                score,missing=_pair_scores(chosen,fitted["users"],fitted["items"],fitted["user_vectors"],fitted["item_vectors"])
                columns[f"{model}_seed{seed}"]=score
        columns[f"{model}_cold"]=missing
        rows.append(pl.DataFrame(columns))
    scores=pl.concat(rows)
    base=pl.read_parquet(config.PROCESSED/"light_features.parquet").select(
        "pair_id","model_end_day","past_return_rate","past_active_days")
    output=base.join(scores,on="pair_id").sort("pair_id")
    output.write_parquet(config.PROCESSED/f"{model}_features.parquet",compression="zstd")
    return output
