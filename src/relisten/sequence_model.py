"""公式SASRecのモデル・サンプラーを使い、CUDA部分だけをMacに合わせる。"""
import json
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from . import config
from .light_models import benchmark_pairs, history_before
from .vendor.sasrec_data import TrainDataset,collate_fn
from .vendor.sasrec_model import SASRecEncoder


PARAMETERS={"max_sequence_length":200,"embedding_dim":64,"num_heads":2,"num_layers":2,"dropout":0.0}


def device_name():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_sasrec(end_day):
    directory=config.PROCESSED/f"models/day_{end_day}"
    directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    events=history_before(end_day).filter(pl.col("played_ratio_pct")>=50).select("uid","item_id","timestamp").sort("uid","timestamp","item_id").collect()
    items=np.sort(events["item_id"].unique().to_numpy())
    events=events.with_columns(pl.Series("index",np.searchsorted(items,events["item_id"].to_numpy())+1))
    sequences=events.group_by("uid",maintain_order=True).agg(pl.col("index").tail(200).alias("item_id")).filter(pl.col("item_id").list.len()>=2)
    loader=DataLoader(TrainDataset(sequences,len(items),200),batch_size=256,shuffle=True,drop_last=True,
                      num_workers=0,pin_memory=False,collate_fn=collate_fn)
    model=SASRecEncoder(num_items=len(items),**PARAMETERS).to(device_name())
    model._encoder.use_nested_tensor=False  # MPS未対応のeval fastpathだけを無効化する。
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    checkpoint=directory/"SASRec_training.pt"
    first_epoch=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device_name(),weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch=saved["epoch"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
    model.train()
    timing_path=config.RESULTS/f"sasrec_training_day_{end_day}.csv"
    times=pl.read_csv(timing_path).filter(pl.col("epoch")<=first_epoch).to_dicts() if first_epoch and timing_path.exists() else []
    for epoch in range(first_epoch,100):
        begin=time.monotonic();losses=[]
        for batch in loader:
            batch={key:value.to(device_name()) for key,value in batch.items()}
            optimizer.zero_grad()
            loss=model(batch)
            if not torch.isfinite(loss):raise FloatingPointError("SASRec loss is non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        duration=time.monotonic()-begin
        times.append({"epoch":epoch+1,"seconds":duration,"mean_loss":float(np.mean(losses))})
        print(f"SASRec day{end_day} epoch{epoch+1}/100: {duration:.1f}s loss={np.mean(losses):.5f}",flush=True)
        if (epoch+1)%10==0:
            temporary=checkpoint.with_suffix(".tmp")
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch+1,
                        "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state()},temporary)
            temporary.replace(checkpoint)
            pl.DataFrame(times).write_csv(timing_path)
    model.to("cpu")
    torch.save(model.state_dict(),directory/"SASRec.pt")
    np.save(directory/"SASRec_items.npy",items)
    sequences.write_parquet(directory/"SASRec_train_sequences.parquet")
    metadata={**PARAMETERS,"end_day":end_day,"max_timestamp":events["timestamp"].max(),"epochs":100,
              "seed":42,"learning_rate":.001,"batch_size":256,"device":device_name(),
              "training_users":len(sequences),"items":len(items),"code_source":"vendor/SOURCE.json"}
    (directory/"SASRec.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return model,items


def _batch_inputs(sequences,device):
    return {"item.ids":torch.tensor([item for sequence in sequences for item in sequence],dtype=torch.long,device=device),
            "item.length":torch.tensor([len(sequence) for sequence in sequences],dtype=torch.long,device=device)}


def _score_sequences(model,sequences,target_indices):
    model._encoder.use_nested_tensor=False
    with torch.inference_mode():
        vectors=model(_batch_inputs(sequences,device_name()))
        targets=model.item_embeddings(torch.tensor(target_indices,dtype=torch.long,device=device_name()))
        scores=(vectors*targets).sum(dim=1)
    return scores.cpu().numpy()


def cold_score_bound(model):
    # 最終LayerNormの出力ノルムとアイテムノルムのCauchy下限。評価データを見ない。
    norm=model._encoder.layers[-1].norm2
    with torch.no_grad():
        user_bound=norm.weight.abs().max()*np.sqrt(PARAMETERS["embedding_dim"])+norm.bias.norm()
        item_bound=model.item_embeddings.weight[1:].norm(dim=1).max()
        return -float((user_bound*item_bound).cpu())-1e-6


def build_sasrec_features():
    pairs=benchmark_pairs()
    history=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").filter(pl.col("played_ratio_pct")>=50).select(
        "uid","item_id",pl.col("timestamp").cast(pl.Int64)).sort("uid","timestamp","item_id").collect()
    histories={h["uid"][0]:h for h in history.partition_by("uid",maintain_order=True)}
    outputs=[]
    for day in (config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY):
        model,items=train_sasrec(day)
        model=model.to(device_name()).eval()
        selected=pairs.filter(pl.col("split")=="train" if day==config.HEAD_REPRESENTATION_END_DAY else pl.col("split")!="train")
        # 起点時の履歴のみ。未学習曲を履歴から除き、対象なら最低スコア。
        floor=cold_score_bound(model)
        scores=np.full(len(selected),floor,dtype=np.float64)
        pending_sequences,pending_targets,pending_positions=[],[],[]
        def flush():
            if not pending_sequences:return
            scores[pending_positions]=_score_sequences(model,pending_sequences,pending_targets)
            pending_sequences.clear();pending_targets.clear();pending_positions.clear()
        for positions in selected.with_row_index("position").partition_by("uid",maintain_order=True):
            uid=positions["uid"][0]
            if uid not in histories:continue
            h=histories[uid]
            mapped=np.searchsorted(items,h["item_id"].to_numpy())
            known=(mapped<len(items))&(items[np.minimum(mapped,len(items)-1)]==h["item_id"].to_numpy())
            times=h["timestamp"].to_numpy()[known]
            mapped=mapped[known]+1
            for row in positions.iter_rows(named=True):
                target=int(np.searchsorted(items,row["item_id"]))
                if target==len(items) or items[target]!=row["item_id"]:continue
                end=np.searchsorted(times,row["anchor_timestamp"],side="left")
                sequence=mapped[max(0,end-200):end].tolist()
                if row["played_ratio_pct"]>=50:sequence=(sequence+[target+1])[-200:]
                if not sequence:continue
                pending_sequences.append(sequence);pending_targets.append(target+1);pending_positions.append(row["position"])
                if len(pending_sequences)==128:flush()
        flush()
        outputs.append(selected.select("pair_id").with_columns(
            pl.Series("sasrec",scores),pl.Series("SASRec_cold",scores==floor)))
        del model
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    base=pl.read_parquet(config.PROCESSED/"light_features.parquet").select("pair_id","model_end_day","past_return_rate","past_active_days")
    features=base.join(pl.concat(outputs),on="pair_id").sort("pair_id")
    features.write_parquet(config.PROCESSED/"SASRec_features.parquet")
    return features
