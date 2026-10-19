"""ACT-Rのbase-levelだけを経路別に替える事前指定の補助分析。"""
import numpy as np
import polars as pl
from scipy.special import softmax

from . import config
from .light_models import benchmark_pairs


def _base_variants(pairs,history,original):
    times=history["timestamp"].to_numpy()
    items=history["item_id"].to_numpy()
    organic=history["is_organic"].to_numpy()==1
    local=np.unique(np.r_[items,pairs["item_id"].to_numpy()])
    indices=np.searchsorted(local,items)
    rows={"pair_id":pairs["pair_id"].to_numpy()}
    for lower in config.SENSITIVITY_LOWER_DAYS:
        rows[f"organic_base_l{lower}"]=[]
        rows[f"recommendation_base_l{lower}"]=[]
    for row in pairs.iter_rows(named=True):
        end=np.searchsorted(times,row["anchor_timestamp"],side="left")
        target=np.searchsorted(local,row["item_id"])
        previous=indices[:end]
        candidates=np.unique(np.r_[previous,target])
        position=np.searchsorted(candidates,target)
        for lower in config.SENSITIVITY_LOWER_DAYS:
            score=original[lower][row["pair_id"]]
            if row["task"]=="T2":
                org_score=rec_score=score  # T2では全版ともbase-levelを使わない。
            else:
                ages=(row["anchor_timestamp"]+lower*config.SECONDS_PER_DAY-times[:end])/config.SECONDS_PER_DAY
                weights=ages**-.5
                base=np.bincount(previous,weights=weights,minlength=len(local))
                org=np.bincount(previous,weights=weights*organic[:end],minlength=len(local))
                base[target]+=lower**-.5  # 今回の推薦再生は全経路版と推薦版に含む。
                rec=base-org
                spreading_partial=score-softmax(base[candidates])[position]
                org_score=spreading_partial+softmax(org[candidates])[position]
                rec_score=spreading_partial+softmax(rec[candidates])[position]
            rows[f"organic_base_l{lower}"].append(float(org_score))
            rows[f"recommendation_base_l{lower}"].append(float(rec_score))
    return pl.DataFrame(rows)


def build_base_variant_features():
    pairs=benchmark_pairs()
    original=pl.read_parquet(config.PROCESSED/"light_features.parquet").sort("pair_id")
    assert original["pair_id"].to_list()==list(range(len(pairs)))
    scores={lower:original[f"actr_l{lower}"].to_numpy() for lower in config.SENSITIVITY_LOWER_DAYS}
    history=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").select(
        "uid","item_id",pl.col("timestamp").cast(pl.Int64),"is_organic").filter(
        pl.col("timestamp")<=pairs["anchor_timestamp"].max()).sort("uid","timestamp","item_id").collect()
    histories={h["uid"][0]:h for h in history.partition_by("uid",maintain_order=True)}
    variants=[]
    groups=pairs.partition_by("uid",maintain_order=True)
    for number,group in enumerate(groups):
        variants.append(_base_variants(group,histories[group["uid"][0]],scores))
        if number%500==0:print(f"ACT-R base variants {number}/{len(groups)} users",flush=True)
    base=original.select("pair_id","model_end_day","past_return_rate","past_active_days")
    result=base.join(pl.concat(variants),on="pair_id").sort("pair_id")
    result.write_parquet(config.PROCESSED/"actr_base_variant_features.parquet")
    return result
