import numpy as np
import polars as pl
import pytest

from relisten.paired_routes import paired_route_auc,seen_a_c_ranks


def example_frame():
    return pl.DataFrame({'pair_id':range(10),'uid':[1,1,1,2,2,3,3,4,4,4],
        'label_b':[0,1,1,1,1,0,1,0,1,1],'label_c':[0,0,1,0,1,0,0,0,0,1],
        's':[0.,2.,1.,0.,1.,0.,1.,3.,2.,1.]})


def test_bc_uses_identical_pairs_and_only_shared_eligible_users():
    frame=example_frame();result,users=paired_route_auc(frame,['s'])
    assert users['uid'].to_list()==[1,4]
    assert result['b_auc_users']==result['c_auc_users']==3
    assert result['paired_users']==2 and result['paired_pairs']==6
    assert result['b_auc_common_users']==.5 and result['c_auc_common_users']==.25
    assert result['mean_auc_difference']==.25
    assert result['difference_ci_low']==0 and result['difference_ci_high']==.5
    assert users['b_n'].equals(users['c_n'])
    assert paired_route_auc(frame.reverse(),['s'])[0]==result
    with pytest.raises(ValueError,match='Duplicate'):paired_route_auc(pl.concat([frame,frame.head(1)]),['s'])


def test_three_seed_auc_mean_precedes_subtraction_and_bootstrap():
    frame=example_frame().with_columns((-100*pl.col('s')).alias('s2'),pl.col('s').alias('s3'))
    result,users=paired_route_auc(frame,['s','s2','s3'])
    assert result['seeds']==3
    assert np.isclose(result['mean_auc_difference'],1/12)
    assert np.allclose(users['b_minus_c'],[1/6,0.])
    assert np.isclose(result['difference_ci_low'],0) and np.isclose(result['difference_ci_high'],1/6)
    # 先にスコアを平均するensembleとは異なる推定量。
    ensemble=frame.with_columns(pl.mean_horizontal('s','s2','s3').alias('ensemble'))
    assert paired_route_auc(ensemble,['ensemble'])[0]['mean_auc_difference']!=result['mean_auc_difference']


def test_seen_rank_test_uses_only_matching_snapshot_and_model_average():
    rows=[]
    for task in ('T1','T2'):
        for selection in ('matching_snapshot','seen_by_both'):
            for model,value in zip(('A','B','C'),(.1,.2,.3) if selection=='matching_snapshot' else (.3,.2,.1)):
                rows.append({'family':model,'task':task,'selection':selection,'split':'test','lower_days':1,
                    'user_auc_after':value,'pairs_after':10,'auc_users_after':2})
    a=pl.DataFrame({'model':['A','A','B','C'],'ndcg_at_10':[0.,.2,.2,.3]})
    tau,detail=seen_a_c_ranks(pl.DataFrame(rows),a)
    assert tau.height==2 and detail.height==6
    assert np.allclose(tau['kendall_tau'],1.)
    assert np.allclose(tau['p_value'],1/3)
    assert detail.filter(pl.col('family')=='A')['a_ndcg_at_10'].to_list()==[.1,.1]
    with pytest.raises(ValueError,match='model sets differ'):seen_a_c_ranks(pl.DataFrame(rows),a.filter(pl.col('model')!='C'))
