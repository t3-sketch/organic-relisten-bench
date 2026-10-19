import polars as pl

from relisten import config,evaluation


def test_t2_diagnostics_leave_primary_predictions_and_protocol_unchanged(tmp_path,monkeypatch):
    processed=tmp_path/'primary';results=tmp_path/'results'
    processed.mkdir();results.mkdir()
    monkeypatch.setattr(config,'PROCESSED',processed)
    monkeypatch.setattr(config,'RESULTS',results)
    monkeypatch.setattr(config,'SENSITIVITY_LOWER_DAYS',(1,7,14))
    d=config.SECONDS_PER_DAY
    pairs=pl.DataFrame({'pair_id':range(8),'uid':[1]*6+[2]*2,'item_id':range(10,18),
        'anchor_timestamp':[30*d]*2+[150*d]*2+[165*d]*2+[30*d]*2,
        'task':['T2']*6+['T1']*2,'split':['train']*2+['validation']*2+['test']*2+['train']*2,
        'label_c':[0,1]*4,'label_b':[0,1]*4})
    monkeypatch.setattr(evaluation,'benchmark_pairs',lambda:pairs)
    monkeypatch.setattr(evaluation,'with_labels',lambda frame,lower:frame)
    features=pl.DataFrame({'pair_id':range(6),'model_end_day':[29]*2+[149]*4,
        'past_return_rate':[.1]*6,'past_active_days':[10.]*6,'score':[0.,1.]*3})
    primary_prediction=processed/'predictions/ACT-R_score_only_T2_test.parquet'
    primary_prediction.parent.mkdir()
    primary_prediction.write_bytes(b'primary sentinel')
    primary_protocol=results/'common_head_protocol.json'
    primary_protocol.write_text('primary protocol sentinel')
    diagnostic=tmp_path/'diagnostic'
    table=evaluation.evaluate_features(features,{'ACT-R':'score'},'actr_component_metrics.csv',
        tasks=('T2',),artifact_root=diagnostic)
    assert set(table['task'])=={'T2'}
    assert primary_prediction.read_bytes()==b'primary sentinel'
    assert primary_protocol.read_text()=='primary protocol sentinel'
    assert (diagnostic/'predictions/ACT-R_score_only_T2_test.parquet').exists()
    assert (diagnostic/'user_auc/ACT-R_score_only_T2_c_test_l1.parquet').exists()
    assert not (processed/'user_auc').exists()
    assert (results/'actr_component_metrics_head_protocol.json').exists()
