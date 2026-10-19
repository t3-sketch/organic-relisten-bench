import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")
from relisten.end_to_end import PairDataset


def test_direct_label_examples_use_prefix_and_current_recommendation_once():
    pairs=pl.DataFrame({"pair_id":[0,1],"uid":[1,1],"item_id":[20,30],"anchor_timestamp":[30,30],
                        "played_ratio_pct":[100,100],"label_c":[1,0]})
    items=np.array([10,20])
    before={1:(np.array([10,20,30]),np.array([1,2,1]))}
    after={1:(np.array([10,20,30,40]),np.array([1,2,1,2]))}
    a=PairDataset(pairs,before,items)
    b=PairDataset(pairs,after,items)
    assert a[0]==b[0]==(0,[1,2,2],2,1)
    assert a[1]==b[1]==(1,[1,2],0,0)
