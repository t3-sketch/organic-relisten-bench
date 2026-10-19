import numpy as np
import polars as pl
import pytest

from relisten import config
from relisten.supplementary import observed_items,seen_pair_flags,subset_statistics,average_seed_auc,combined_rank_table
from relisten.metrics import user_auc_table


def test_vocab_cutoffs_and_matching_snapshot_do_not_use_future_items():
    d=86400
    events=pl.DataFrame({'item_id':[1,2,3,4,5],'timestamp':[0,30*d-1,30*d,150*d-1,150*d]})
    vocab={day:observed_items(events,day)['item_id'] for day in (29,149)}
    assert vocab[29].to_list()==[1,2]
    assert vocab[149].to_list()==[1,2,3,4]
    pairs=pl.DataFrame({'pair_id':range(4),'item_id':[2,3,3,5],'split':['train','train','test','test'],
        'anchor_timestamp':[31*d,31*d,165*d,165*d]})
    flags=seen_pair_flags(pairs,vocab)
    assert flags['seen_in_training'].to_list()==[True,False,True,False]
    assert flags['seen_in_both_training'].to_list()==[True,False,False,False]
    future=pl.concat([events,pl.DataFrame({'item_id':[3,5],'timestamp':[200*d,201*d]})])
    later={day:observed_items(future,day)['item_id'] for day in (29,149)}
    assert seen_pair_flags(pairs,later).equals(flags)


def test_restricted_auc_and_fixed_minimum_fraction_not_recomputed_after_filter():
    frame=pl.DataFrame({'uid':[1,1,1,2,2],'label_c':[0,1,0,0,1],
        's':[-10.,1.,2.,3.,4.],'seen_in_training':[False,True,True,True,False],'cold':[True,False,False,False,False]})
    stats,auc=subset_statistics(frame,'s','cold')
    assert stats['minimum_fraction_before']==.2 and stats['minimum_fraction_after']==0
    assert stats['fallback_fraction_before']==.2 and stats['fallback_fraction_after']==0
    assert stats['pairs_after']==3
    assert auc['uid'].to_list()==[1] and auc['auc'].to_list()==[0.]


def test_seed_average_preserves_user_pairing_and_rejects_different_labels():
    first=user_auc_table([1,1,2,2],[0,1,0,1],[0.,1.,0.,1.])
    second=user_auc_table([1,1,2,2],[0,1,0,1],[1.,0.,0.,1.])
    assert average_seed_auc([first,second])['auc'].to_list()==[.5,1.]
    bad=second.with_columns(pl.lit(2).alias('positives'))
    with pytest.raises(ValueError,match='Seed cohorts differ'):average_seed_auc([first,bad])


def test_rank_summary_both_tasks_raw_only_tau_b_with_ties(monkeypatch):
    monkeypatch.setattr(config,'BENCHMARK_MODELS',('A','B','C'))
    rows=[]
    for task in ('T2','T1'):
        for model,b,c in zip(('A','B','C'),(.2,.3,.4),(.4,.3,.2) if task=='T2' else (.2,.3,.4)):
            for route,value in [('B',b),('C',c)]:
                rows.append({'family':model,'task':task,'route':route,'user_auc':value,'protocol':'raw_score','lower_days':1,'split':'test'})
    results=pl.DataFrame(rows)
    task_a=pl.DataFrame({'model':['A','A','B','C'],'seed':[41,42,42,42],'ndcg_at_10':[.1,.3,.2,.4]})
    table=combined_rank_table(results,task_a)
    assert table.height==6
    assert np.allclose(table.filter(pl.col('task')=='T2')['tau_a_c'],-np.sqrt(2/3))
    assert np.allclose(table.filter(pl.col('task')=='T1')['tau_b_c'],1.)
    assert table.filter(pl.col('family')=='A')['a_ndcg_at_10'].to_list()==[.2,.2]
    with pytest.raises(ValueError,match='model sets differ'):combined_rank_table(results,task_a.filter(pl.col('model')!='C'))
