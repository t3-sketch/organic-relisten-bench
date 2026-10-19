"""補助実験：29日目SASRecから、T2のCラベルで全層を直接学習する。"""
import json
import time

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from . import config
from .labels import with_labels
from .light_models import benchmark_pairs
from .metrics import evaluate_probabilities
from .sequence_model import PARAMETERS,_batch_inputs,device_name
from .vendor.sasrec_model import SASRecEncoder


EPOCHS=10  # 補助実験の事前固定値。検証・テストで選ばない。


class PairDataset(Dataset):
    def __init__(self,pairs,histories,items):
        self.rows=pairs.select("pair_id","uid","item_id","anchor_timestamp","played_ratio_pct","label_c").rows()
        self.histories=histories
        self.items=items

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        pair_id,uid,item,timestamp,ratio,label=self.rows[index]
        times,mapped=self.histories.get(uid,(np.array([],dtype=np.int64),np.array([],dtype=np.int64)))
        end=np.searchsorted(times,timestamp,side="left")
        sequence=mapped[max(0,end-200):end].tolist()
        target=np.searchsorted(self.items,item)
        known=target<len(self.items) and self.items[target]==item
        if known and ratio>=50:sequence=(sequence+[int(target)+1])[-200:]
        # 空履歴は既存の0埋め込み1個。未知対象は学習済み曲より必ず低い固定スコア。
        return pair_id,sequence or [0],int(target)+1 if known else 0,label


def _collate(rows):return tuple(zip(*rows))


class DirectRelistening(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.log_scale=nn.Parameter(torch.tensor(0.54132485))  # softplus=1
        self.bias=nn.Parameter(torch.tensor(0.0))

    def forward(self,sequences,targets):
        device=device_name()
        inputs=_batch_inputs(sequences,device)
        encoded,mask=self.encoder._apply_sequential_encoder(inputs["item.ids"],inputs["item.length"])
        users=self.encoder._get_last_embedding(encoded,mask)
        target=torch.tensor(targets,dtype=torch.long,device=device)
        scores=torch.tanh((users*self.encoder.item_embeddings(target)).sum(dim=1))
        scores=torch.where(target>0,scores,torch.full_like(scores,-1.000001))
        return F.softplus(self.log_scale)*scores+self.bias


def _histories(items,max_timestamp):
    events=pl.scan_parquet(config.RAW/"flat/50m/listens.parquet").filter(
        (pl.col("played_ratio_pct")>=50)&(pl.col("timestamp")<max_timestamp)).select(
        "uid","item_id",pl.col("timestamp").cast(pl.Int64)).sort("uid","timestamp","item_id").collect()
    result={}
    for h in events.partition_by("uid",maintain_order=True):
        mapped=np.searchsorted(items,h["item_id"].to_numpy())
        known=(mapped<len(items))&(items[np.minimum(mapped,len(items)-1)]==h["item_id"].to_numpy())
        result[h["uid"][0]]=(h["timestamp"].to_numpy()[known],mapped[known]+1)
    return result


def run_end_to_end():
    directory=config.PROCESSED/f"models/day_{config.HEAD_REPRESENTATION_END_DAY}"
    items=np.load(directory/"SASRec_items.npy")
    pairs=benchmark_pairs().filter(pl.col("task")=="T2")
    histories=_histories(items,pairs["anchor_timestamp"].max())
    rows=[]
    for lower in config.SENSITIVITY_LOWER_DAYS:
        torch.manual_seed(42);np.random.seed(42)
        torch.set_num_threads(4)
        encoder=SASRecEncoder(num_items=len(items),**PARAMETERS)
        encoder.load_state_dict(torch.load(directory/"SASRec.pt",map_location="cpu",weights_only=True))
        encoder._encoder.use_nested_tensor=False
        model=DirectRelistening(encoder).to(device_name())
        cohort=with_labels(pairs,lower)
        training=cohort.filter(pl.col("split")=="train")
        loader=DataLoader(PairDataset(training,histories,items),batch_size=256,shuffle=True,
                          drop_last=False,num_workers=0,collate_fn=_collate)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        checkpoint=config.PROCESSED/f"models/SASRec_e2e_l{lower}.pt"
        first_epoch=0;times=[]
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location=device_name(),weights_only=False)
            model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
            first_epoch=saved["epoch"];times=saved["times"]
            torch.set_rng_state(saved["torch_rng"].cpu())
        for epoch in range(first_epoch,EPOCHS):
            model.train();begin=time.monotonic();loss_sum=0.;count=0
            for pair_ids,sequences,targets,labels in loader:
                optimizer.zero_grad()
                logits=model(sequences,targets)
                labels=torch.tensor(labels,dtype=torch.float32,device=device_name())
                loss=F.binary_cross_entropy_with_logits(logits,labels)
                if not torch.isfinite(loss):raise FloatingPointError("End-to-End loss is non-finite")
                loss.backward();optimizer.step()
                loss_sum+=float(loss.detach().cpu())*len(labels);count+=len(labels)
            times.append({"epoch":epoch+1,"seconds":time.monotonic()-begin,"loss":loss_sum/count})
            print(f"SASRec End-to-End L={lower} epoch{epoch+1}/{EPOCHS}: {times[-1]}",flush=True)
            temporary=checkpoint.with_suffix(".tmp")
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch+1,
                        "times":times,"torch_rng":torch.get_rng_state()},temporary)
            temporary.replace(checkpoint)
            pl.DataFrame(times).write_csv(config.RESULTS/f"sasrec_e2e_training_l{lower}.csv")
        model.eval()
        for split in ("validation","test"):
            evaluation=cohort.filter(pl.col("split")==split)
            probabilities=[]
            loader=DataLoader(PairDataset(evaluation,histories,items),batch_size=128,shuffle=False,
                              num_workers=0,collate_fn=_collate)
            with torch.inference_mode():
                for pair_ids,sequences,targets,labels in loader:
                    probabilities.extend(model(sequences,targets).sigmoid().cpu().numpy())
            metrics,_=evaluate_probabilities(evaluation["uid"],evaluation["label_c"],np.array(probabilities))
            rows.append({"model":"SASRec-End-to-End","protocol":"direct_label_training","task":"T2","route":"C",
                         "lower_days":lower,"split":split,"is_main_lower":lower==config.LOWER_BOUND_DAYS,**metrics})
            evaluation.select("pair_id","uid","label_c").with_columns(pl.Series("probability",probabilities)).write_parquet(
                config.PROCESSED/f"SASRec_e2e_{split}_l{lower}.parquet")
        del model,encoder,optimizer
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    table=pl.DataFrame(rows)
    table.write_csv(config.RESULTS/"sasrec_end_to_end_metrics.csv")
    (config.RESULTS/"sasrec_end_to_end_protocol.json").write_text(json.dumps({
        "task":"T2","route":"C","lower_days":list(config.SENSITIVITY_LOWER_DAYS),"epochs":EPOCHS,
        "initial_model_day":config.HEAD_REPRESENTATION_END_DAY,"label_training_days":[30,59],
        "all_training_labels_mature_by_day":149,"evaluation_input":"strict pre-anchor eligible history + current recommendation once",
        "score_link":"positive_scale * tanh(raw dot) + bias; cold score=-1.000001 before scale",
        "tuning":"none;10 epochs,Adam0.001,batch256,seed42 fixed before this experiment",
        "limitation":"29-day initialized model is used at evaluation too; main common-head uses day149. Difference includes representation-age effects and cannot isolate objective alone",
    },indent=2)+"\n")
    return table
