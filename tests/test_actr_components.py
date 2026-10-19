import numpy as np
import polars as pl
from scipy import sparse

from relisten import config
from relisten.actr_components import _user_components
from relisten.light_models import _user_audio_actr,fit_item_sessions


def test_components_reconstruct_original_and_organic_input_ignores_recommendations():
    d=config.SECONDS_PER_DAY
    pairs=pl.DataFrame({"pair_id":[0],"uid":[1],"item_id":[30],"anchor_timestamp":[30*d],"task":["T2"]})
    history=pl.DataFrame({"uid":[1,1],"item_id":[10,20],"timestamp":[d,2*d],"is_organic":[1,0]})
    items=np.array([10,20,30,40])
    audio=np.array([[1.,0.],[0.,1.],[1.,0.],[1.,0.]],dtype=np.float32)
    sessions=sparse.csr_matrix([[1,0],[1,1],[0,1],[1,0]],dtype=np.float32)
    prevalence=np.asarray(sessions.sum(axis=1)).ravel()
    model=(items,sessions,prevalence)
    components=_user_components(pairs,history,items,audio,model,model)
    original=_user_audio_actr(pairs,history,items,audio,items,sessions,prevalence,2)
    assert np.isclose(components['s_only'][0]+components['p_only'][0],original['actr_l1'][0])
    assert components['p_raw'][0]==0 and original['audio_cosine'][0]==1
    future=pl.concat([history,pl.DataFrame({"uid":[1],"item_id":[40],"timestamp":[40*d],"is_organic":[1]})])
    assert components.rows()==_user_components(pairs,future,items,audio,model,model).rows()
    another_rec=pl.concat([history,pl.DataFrame({"uid":[1],"item_id":[40],"timestamp":[20*d],"is_organic":[0]})])
    changed=_user_components(pairs,another_rec,items,audio,model,model)
    assert components['organic_sp'][0]==changed['organic_sp'][0]
    assert components['p_raw'][0]!=changed['p_raw'][0]


def test_rare_context_self_lift_can_saturate_spreading():
    pairs=pl.DataFrame({"pair_id":[0],"item_id":[30],"anchor_timestamp":[30]})
    history=pl.DataFrame({"item_id":[10],"timestamp":[10],"is_organic":[1]})
    items=np.array([10,30])
    audio=np.eye(2,dtype=np.float32)
    sessions=sparse.csr_matrix(([1.,1.],([0,1],[0,1])),shape=(2,1000),dtype=np.float32)
    model=(items,sessions,np.array([1.,1.]))
    result=_user_components(pairs,history,items,audio,model,model)
    assert result['s_only'][0]==0 and result['s_self_mass'][0]==1


def test_organic_sessions_filter_before_gaps_and_obey_model_deadline(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'RAW',tmp_path)
    path=tmp_path/'flat/50m/listens.parquet';path.parent.mkdir(parents=True)
    pl.DataFrame({'uid':[1,1,1,1],'item_id':[10,20,30,40],
        'timestamp':[0,1200,2400,30*config.SECONDS_PER_DAY],'is_organic':[1,0,1,1]}).write_parquet(path)
    all_items,all_sessions=fit_item_sessions(29)
    org_items,org_sessions=fit_item_sessions(29,organic_only=True)
    assert all_items.tolist()==[10,20,30] and all_sessions.shape==(3,1)
    assert org_items.tolist()==[10,30] and org_sessions.shape==(2,2)
    assert (org_sessions@org_sessions.T).toarray()[0,1]==0
