import numpy as np
import polars as pl
import pytest

optuna=pytest.importorskip("optuna")
from relisten import official_models as models


def test_interrupted_search_keeps_completed_trials_and_final_models(tmp_path,monkeypatch):
    monkeypatch.setattr(models.config,"PROCESSED",tmp_path/"processed")
    monkeypatch.setattr(models.config,"RESULTS",tmp_path)
    events=pl.DataFrame({"uid":[1,2],"item_id":[10,20],"timestamp":[0,1]})
    monkeypatch.setattr(models,"_split_for_tuning",lambda day:(events,events,events))
    monkeypatch.setattr(models,"full_catalog_ndcg",lambda *args,**kwargs:.8)
    calls=[]
    def fit(model,matrix,parameters,seed):
        calls.append((dict(parameters),seed))
        if len(calls)==3:raise KeyboardInterrupt()
        return np.ones((2,2)),np.ones((2,2))
    monkeypatch.setattr(models,"_matrix_fit",fit)
    with pytest.raises(KeyboardInterrupt):models.fit_matrix_factorization("iALS",29)
    study=optuna.load_study(study_name="iALS",storage=f"sqlite:///{tmp_path/'processed/models/day_29/iALS_study.sqlite3'}")
    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    assert len(completed)==2
    initial=[t.params for t in completed]
    result=models.fit_matrix_factorization("iALS",29)
    trials=pl.read_csv(tmp_path/"iALS_tuning_day_29.csv")
    assert trials.filter(pl.col("state")=="COMPLETE").height==10
    study=optuna.load_study(study_name="iALS",storage=f"sqlite:///{tmp_path/'processed/models/day_29/iALS_study.sqlite3'}")
    assert [t.params for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE][:2]==initial
    expected=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    for completed in (t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE):
        trial=expected.ask()
        assert models._suggest_parameters(trial,"iALS")==completed.params
        expected.tell(trial,completed.value)
    assert result["seeds"]==models.FINAL_SEEDS
    count=len(calls)
    models.fit_matrix_factorization("iALS",29)
    assert len(calls)==count


def test_database_commit_without_callback_still_restores_next_draw():
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    for _ in range(6):
        trial=study.ask();models._suggest_parameters(trial,"BPR");study.tell(trial,.8)
    committed=study.ask();models._suggest_parameters(committed,"BPR");study.tell(committed,.8)
    expected=study.ask()
    expected_parameters=models._suggest_parameters(expected,"BPR")
    study.tell(expected,state=optuna.trial.TrialState.FAIL)
    study.sampler=models._replayed_sampler(study,"BPR")
    resumed=study.ask()
    assert models._suggest_parameters(resumed,"BPR")==expected_parameters
