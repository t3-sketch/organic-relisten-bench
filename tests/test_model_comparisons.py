import polars as pl
import pytest

from relisten import config
from relisten.analysis import paired_model_comparisons


def test_all_model_gate_and_same_user_auc_differences(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'PROCESSED',tmp_path)
    monkeypatch.setattr(config,'RESULTS',tmp_path)
    monkeypatch.setattr(config,'BENCHMARK_MODELS',('AudioCosine','MostPop'))
    predictions=tmp_path/'predictions';predictions.mkdir()
    (tmp_path/'raw_scores/predictions').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='waiting for'):paired_model_comparisons()
    (tmp_path/'light_model_metrics.csv').write_text('completed fixture\n')
    (tmp_path/'raw_score_model_metrics.csv').touch()
    for task in ('T2','T1'):
        for protocol in ('raw_score','score_only','baseline_plus_score'):
            for model in config.BENCHMARK_MODELS:
                pl.DataFrame({'pair_id':[0,1,2,3],'uid':[1,1,2,2],'label_c':[0,1,0,1],
                    ('score' if protocol=='raw_score' else 'probability'):[.1,.9,.1,.9] if model=='AudioCosine' else [.5,.5,.5,.5]}).write_parquet(
                    (tmp_path/'raw_scores/predictions' if protocol=='raw_score' else predictions)/f'{model}_{protocol}_{task}_test.parquet')
    result=paired_model_comparisons()
    assert result.height==6
    assert result['paired_users'].to_list()==[2]*6
    assert result['left_model'].to_list()==['AudioCosine']*6
    assert result['mean_auc_difference'].to_list()==[.5]*6
    assert result['difference_ci_low'].to_list()==result['difference_ci_high'].to_list()==[.5]*6
    assert (tmp_path/'top_model_paired_differences.csv').exists()
    path=predictions/'MostPop_score_only_T2_test.parquet'
    pl.read_parquet(path).with_columns((1-pl.col('label_c')).alias('label_c')).write_parquet(path)
    with pytest.raises(ValueError,match='cohorts differ'):paired_model_comparisons()


def test_seed_auc_is_averaged_before_paired_bootstrap(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'PROCESSED',tmp_path)
    monkeypatch.setattr(config,'RESULTS',tmp_path)
    monkeypatch.setattr(config,'BENCHMARK_MODELS',('AudioCosine','BPR'))
    predictions=tmp_path/'predictions';predictions.mkdir()
    (tmp_path/'raw_scores/predictions').mkdir(parents=True)
    for name in ('light_model_metrics.csv','bpr_model_metrics.csv','raw_score_model_metrics.csv'):(tmp_path/name).touch()
    probabilities={'AudioCosine':[.1,.9,.1,.9],
        'BPR_seed41':[.49,.51,.49,.51],'BPR_seed42':[.9,.1,.49,.51],'BPR_seed43':[.49,.51,.9,.1]}
    for task in ('T2','T1'):
        for protocol in ('raw_score','score_only','baseline_plus_score'):
            for name,scores in probabilities.items():
                pl.DataFrame({'pair_id':[0,1,2,3],'uid':[1,1,2,2],'label_c':[0,1,0,1],
                    ('score' if protocol=='raw_score' else 'probability'):scores}).write_parquet((tmp_path/'raw_scores/predictions' if protocol=='raw_score' else predictions)/f'{name}_{protocol}_{task}_test.parquet')
    result=paired_model_comparisons()
    assert all(abs(x-1/3)<1e-12 for x in result['mean_auc_difference'])
    assert all(abs(x-1/3)<1e-12 for x in result['difference_ci_low'])
    (predictions/'BPR_seed43_score_only_T2_test.parquet').unlink()
    with pytest.raises(RuntimeError,match='waiting for'):paired_model_comparisons()


def test_sansa_without_an_explicit_resource_status_is_still_pending(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'PROCESSED',tmp_path)
    monkeypatch.setattr(config,'RESULTS',tmp_path)
    monkeypatch.setattr(config,'BENCHMARK_MODELS',('SANSA',))
    with pytest.raises(RuntimeError,match='sansa_model_metrics.csv'):paired_model_comparisons()
