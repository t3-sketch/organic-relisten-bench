import numpy as np
import polars as pl
from scipy import sparse

from relisten import config
from relisten.light_models import _user_audio_actr
from relisten.actr_analysis import _base_variants
from scipy.special import softmax


def test_later_history_does_not_change_audio_or_actr_features():
    d=config.SECONDS_PER_DAY
    pairs=pl.DataFrame({"pair_id":[0,1],"uid":[1,1],"item_id":[30,50],"anchor_timestamp":[30*d,30*d],"task":["T2","T2"]})
    history=pl.DataFrame({"uid":[1,1,1],"item_id":[10,20,40],"timestamp":[d,2*d,40*d],"is_organic":[1,1,1]})
    items=np.array([10,20,30,40])
    audio=np.array([[1.,0.],[1.,0.],[1.,0.],[0.,1.]],dtype=np.float32)
    sessions=sparse.csr_matrix([[1,0],[1,1],[0,1],[1,0]],dtype=np.float32)
    prevalence=np.asarray(sessions.sum(axis=1)).ravel()
    before=_user_audio_actr(pairs,history.head(2),items,audio,items,sessions,prevalence,2)
    after=_user_audio_actr(pairs,history,items,audio,items,sessions,prevalence,2)
    assert before.rows()==after.rows()
    assert before["audio_cosine"][0]==1.0
    assert before["audio_missing"].to_list()==[False,True]
    assert before["actr_l1"][0]==before["actr_l7"][0]==before["actr_l14"][0]


def test_route_base_variants_use_only_past_and_count_current_rec_once():
    d=config.SECONDS_PER_DAY
    pairs=pl.DataFrame({"pair_id":[0,1],"item_id":[10,30],"anchor_timestamp":[30*d,30*d],"task":["T1","T2"]})
    past=pl.DataFrame({"item_id":[10,20],"timestamp":[28*d,29*d],"is_organic":[1,0]})
    original={}
    for lower in config.SENSITIVITY_LOWER_DAYS:
        base=np.array([(lower+2)**-.5+lower**-.5,(lower+1)**-.5])
        original[lower]=np.array([.2+softmax(base)[0],.4])
    result=_base_variants(pairs,past,original)
    future=pl.concat([past,pl.DataFrame({"item_id":[10],"timestamp":[31*d],"is_organic":[1]})])
    assert result.rows()==_base_variants(pairs,future,original).rows()
    for lower in config.SENSITIVITY_LOWER_DAYS:
        assert np.isclose(result[f"organic_base_l{lower}"][0],.2+softmax([(lower+2)**-.5,0])[0])
        assert np.isclose(result[f"recommendation_base_l{lower}"][0],.2+softmax([lower**-.5,(lower+1)**-.5])[0])
        assert result[f"organic_base_l{lower}"][1]==result[f"recommendation_base_l{lower}"][1]==.4
