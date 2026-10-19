import numpy as np
import polars as pl

from relisten.metrics import evaluate_ranking
from relisten.score_protocol import recent_play_counts,training_user_percentiles


def test_raw_ranking_does_not_flip_a_negative_model_direction():
    result,auc=evaluate_ranking([1,1,2,2],[0,1,0,1],[10.,-5.,20.,-2.],n_bootstrap=100)
    assert result['user_auc']==0
    assert result['global_auc']==0
    assert 'log_loss' not in result and 'brier' not in result
    assert auc['auc'].to_list()==[0.,0.]


def test_recent_7d_includes_left_boundary_excludes_anchor_and_future():
    d=86400
    pairs=pl.DataFrame({'pair_id':[0,1,2],'uid':[1,1,2],'anchor_timestamp':[10*d,11*d,10*d]})
    history=pl.DataFrame({'uid':[1]*7,'timestamp':[3*d-1,3*d,4*d,10*d-1,10*d,10*d,12*d]})
    assert recent_play_counts(pairs,history)['recent_7d_plays'].to_list()==[3,4,0]
    altered=pl.concat([history,pl.DataFrame({'uid':[1,2],'timestamp':[20*d,20*d]})])
    assert recent_play_counts(pairs,history).equals(recent_play_counts(pairs,altered))


def test_training_percentile_midrank_fallback_and_future_batch_invariance():
    train=pl.DataFrame({'uid':[1,1,1,2],'s':[0.,2.,2.,10.]})
    frame=pl.DataFrame({'uid':[1,1,2,3],'s':[2.,1.,10.,2.]})
    values,fallback=training_user_percentiles(train,frame,'s')
    assert np.allclose(values,[2/3,1/3,.5,.5])
    assert fallback==1
    extended=pl.concat([frame,pl.DataFrame({'uid':[1,3],'s':[999.,-999.]})])
    assert np.array_equal(training_user_percentiles(train,extended,'s')[0][:4],values)


def test_full_protocol_keeps_raw_direction_and_auxiliary_separate(tmp_path,monkeypatch):
    from relisten import config,evaluation,score_protocol,reporting,analysis
    processed=tmp_path/'processed';processed.mkdir()
    results=tmp_path/'results';results.mkdir()
    monkeypatch.setattr(config,'PROCESSED',processed)
    monkeypatch.setattr(config,'RESULTS',results)
    monkeypatch.setattr(config,'SENSITIVITY_LOWER_DAYS',(1,7,14))
    monkeypatch.setattr(config,'BENCHMARK_MODELS',('MostPop',))
    monkeypatch.setattr(reporting,'MODEL_FILES',{'MostPop':'light_model_metrics.csv'})
    monkeypatch.setattr(score_protocol,'SCORE_SPECS',[('tiny.parquet',{'MostPop':'tiny'})])
    d=86400
    pairs=pl.DataFrame({'pair_id':range(12),'uid':[1]*12,'item_id':range(12),
        'anchor_timestamp':[30*d]*2+[150*d]*2+[165*d]*2+[30*d]*2+[150*d]*2+[165*d]*2,
        'task':['T2']*6+['T1']*6,'split':['train']*2+['validation']*2+['test']*2+['train']*2+['validation']*2+['test']*2,
        'label_c':[0,1]*6,'label_b':[0,1]*6})
    features=pl.DataFrame({'pair_id':range(12),'model_end_day':[29]*2+[149]*4+[29]*2+[149]*4,
        'past_return_rate':[.1]*12,'past_active_days':[10.]*12,'tiny':[5.,-2.]*6,'recent_7d_plays':[0.,1.]*6})
    for module in (evaluation,score_protocol):
        monkeypatch.setattr(module,'benchmark_pairs',lambda:pairs)
        monkeypatch.setattr(module,'with_labels',lambda frame,lower:frame)
    monkeypatch.setattr(score_protocol,'build_recent_features',lambda _:features)
    features.write_parquet(processed/'tiny.parquet')
    evaluation.evaluate_features(features,{'MostPop':'tiny'},'light_model_metrics.csv')
    raw=score_protocol.run_score_protocol()
    assert set(raw['model'])=={'MostPop'}
    assert raw['user_auc'].to_list()==[0.]*len(raw)
    coefficients=pl.read_csv(results/'model_head_coefficients.csv')
    assert set(coefficients.filter((pl.col('model')=='MostPop')&pl.col('is_model_score'))['sign'])=={'negative'}
    assert set(pl.read_csv(results/'recent7d_raw_metrics.csv')['model'])=={'Recent7dPlays'}
    assert raw.equals(score_protocol.run_score_protocol())
    (results/'figures').mkdir()
    aggregated=analysis.collect_model_results()
    reporting.write_benchmark_report(aggregated)
    score_protocol.write_score_protocol_report(aggregated)
    assert 'MostPop' in (results/'SCORE_PROTOCOL_REPORT.md').read_text()
    assert '生スコアAUC（主）' in (results/'BENCHMARK_REPORT.md').read_text()
