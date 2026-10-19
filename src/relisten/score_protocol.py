"""設計上の向きの生スコアを主評価にする。表現モデルは再学習しない。"""
import json

import numpy as np
import polars as pl

from . import config
from .evaluation import BASELINE_COLUMNS,common_head,evaluate_features
from .labels import with_labels
from .light_models import benchmark_pairs
from .metrics import evaluate_ranking,evaluate_probabilities
from .official_models import FINAL_SEEDS


RAW_ROOT='raw_scores'
SCORE_SPECS=[('light_features.parquet',{'MostPop':'mostpop','DecayPop':'decaypop','ACT-R':'actr','AudioCosine':'audio_cosine'}),
    ('ItemKNN_features.parquet',{'ItemKNN':'itemknn'}),
    *[(f'{model}_features.parquet',{f'{model}_seed{s}':f'{model}_seed{s}' for s in FINAL_SEEDS}) for model in ('iALS','BPR')],
    ('SASRec_features.parquet',{'SASRec':'sasrec'})]


def recent_play_counts(pairs,history):
    """ユーザーの全再生イベント数。左端を含み、起点の同時刻をすべて除外。"""
    groups={g['uid'][0]:np.sort(g['timestamp'].to_numpy()) for g in history.partition_by('uid')}
    rows=[]
    for group in pairs.partition_by('uid',maintain_order=True):
        times=groups.get(group['uid'][0],np.array([],dtype=np.int64))
        anchors=group['anchor_timestamp'].to_numpy()
        counts=np.searchsorted(times,anchors,side='left')-np.searchsorted(times,anchors-7*config.SECONDS_PER_DAY,side='left')
        rows.append(group.select('pair_id').with_columns(pl.Series('recent_7d_plays',counts)))
    return pl.concat(rows).sort('pair_id')


def build_recent_features(pairs):
    path=config.PROCESSED/'recent7d_features.parquet'
    if path.exists():return pl.read_parquet(path)
    history=pl.scan_parquet(config.RAW/'flat/50m/listens.parquet').select('uid',pl.col('timestamp').cast(pl.Int64)).filter(
        pl.col('timestamp')<pairs['anchor_timestamp'].max()).collect()
    base=pl.read_parquet(config.PROCESSED/'light_features.parquet').select('pair_id','model_end_day',*BASELINE_COLUMNS)
    features=base.join(recent_play_counts(pairs,history),on='pair_id').sort('pair_id')
    features.write_parquet(path)
    return features


def training_user_percentiles(train,frame,column):
    """学習だけでCDFを固定。同点はmidrank、未学習ユーザーは学習全体へ。"""
    groups={g['uid'][0]:np.sort(g[column].to_numpy()) for g in train.partition_by('uid')}
    pooled=np.sort(train[column].to_numpy())
    if not len(pooled):raise ValueError('Percentile requires training scores')
    rows=[];fallback=0
    for group in frame.with_row_index('_position').partition_by('uid',maintain_order=True):
        reference=groups.get(group['uid'][0])
        if reference is None:reference=pooled;fallback+=len(group)
        values=group[column].to_numpy()
        percentile=(np.searchsorted(reference,values,side='left')+np.searchsorted(reference,values,side='right'))/(2*len(reference))
        rows.append(group.select('_position').with_columns(pl.Series('percentile',percentile)))
    return pl.concat(rows).sort('_position')['percentile'].to_numpy(),fallback


def _save_auc(root,name,protocol,task,route,split,lower,auc):
    directory=root/'user_auc';directory.mkdir(parents=True,exist_ok=True)
    auc.write_parquet(directory/f'{name}_{protocol}_{task}_{route}_{split}_l{lower}.parquet')


def _save_predictions(root,name,protocol,task,split,frame,column,values):
    directory=root/'predictions';directory.mkdir(parents=True,exist_ok=True)
    frame.select('pair_id','uid','label_c').with_columns(pl.Series(column,values)).write_parquet(directory/f'{name}_{protocol}_{task}_{split}.parquet')


def run_score_protocol():
    outputs=['raw_score_model_metrics.csv','recent7d_raw_metrics.csv','model_head_coefficients.csv','percentile_head_metrics.csv','recent7d_head_metrics.csv']
    if (config.RESULTS/'score_protocol_completed.json').exists() and all((config.RESULTS/name).exists() for name in outputs):
        return pl.read_csv(config.RESULTS/outputs[0])
    from .reporting import MODEL_FILES
    missing=[name for name in set(MODEL_FILES.values()) if not (config.RESULTS/name).exists()]
    if missing:raise RuntimeError('Raw-score evaluation is waiting for: '+', '.join(missing))
    pairs=benchmark_pairs()
    recent=build_recent_features(pairs)
    evaluate_features(recent,{'Recent7dPlays':'recent_7d_plays'},'recent7d_head_metrics.csv',artifact_root=config.PROCESSED/'recent7d_diagnostics')
    raw_rows=[];diagnostic_rows=[];coefficients=[];percentile_rows=[]
    for filename,models in [*SCORE_SPECS,('recent7d_features.parquet',{'Recent7dPlays':'recent_7d_plays'})]:
        features=recent if filename=='recent7d_features.parquet' else pl.read_parquet(config.PROCESSED/filename)
        frame=pairs.join(features,on='pair_id')
        if frame.height!=pairs.height or frame['pair_id'].n_unique()!=pairs.height:raise ValueError('Scores must cover exactly the benchmark pairs')
        expected=pl.when(pl.col('split')=='train').then(config.HEAD_REPRESENTATION_END_DAY).otherwise(config.REPRESENTATION_END_DAY)
        assert frame.select((pl.col('model_end_day')==expected).all()).item()
        for model,column in models.items():
            diagnostic=model=='Recent7dPlays'
            root=config.PROCESSED/('recent7d_diagnostics' if diagnostic else RAW_ROOT)
            head_root=config.PROCESSED/('recent7d_diagnostics' if diagnostic else '')
            for lower in config.SENSITIVITY_LOWER_DAYS:
                score=f'{column}_l{lower}' if f'{column}_l{lower}' in features.columns else column
                labeled=with_labels(frame,lower)
                for task in ('T2','T1'):
                    cohort=labeled.filter(pl.col('task')==task)
                    train=cohort.filter(pl.col('split')=='train')
                    for route in ('c','b'):
                        y=train[f'label_{route}'].to_numpy()
                        for split in ('validation','test'):
                            evaluation=cohort.filter(pl.col('split')==split)
                            values=evaluation[score].to_numpy()
                            result,auc=evaluate_ranking(evaluation['uid'],evaluation[f'label_{route}'],values)
                            key={'model':model,'protocol':'raw_score','task':task,'route':route.upper(),'split':split,
                                'lower_days':lower,'is_main_lower':lower==config.LOWER_BOUND_DAYS}
                            (diagnostic_rows if diagnostic else raw_rows).append({**key,**result})
                            _save_auc(root,model,'raw_score',task,route,split,lower,auc)
                            if lower==config.LOWER_BOUND_DAYS and route=='c':
                                _save_predictions(root,model,'raw_score',task,split,evaluation,'score',values)
                        for protocol,columns in [('score_only',[score]),('baseline_plus_score',BASELINE_COLUMNS+[score])]:
                            x=train.select(columns).to_numpy()
                            _,head=common_head(x,y,x[:1])
                            coefs=head.named_steps['logisticregression'].coef_[0]
                            for feature,value in zip(columns,coefs):
                                coefficients.append({'model':model,'protocol':protocol,'task':task,'route':route.upper(),
                                    'lower_days':lower,'feature':feature,'is_model_score':feature==score,
                                    'scaled_coefficient':float(value),'sign':'positive' if value>0 else 'negative' if value<0 else 'zero',
                                    'train_pairs':len(train),'representation_end_day':config.HEAD_REPRESENTATION_END_DAY})
                            if lower==config.LOWER_BOUND_DAYS and route=='c':
                                evaluation=cohort.filter(pl.col('split')=='test')
                                probability=head.predict_proba(evaluation.select(columns).to_numpy())[:,1]
                                saved=pl.read_parquet(head_root/'predictions'/f'{model}_{protocol}_{task}_test.parquet').sort('pair_id')
                                reconstructed=evaluation.select('pair_id').with_columns(pl.Series('probability',probability)).sort('pair_id')
                                assert saved['pair_id'].equals(reconstructed['pair_id'])
                                assert np.allclose(saved['probability'],reconstructed['probability'],rtol=1e-7,atol=1e-9),'Refit head differs from saved secondary predictions'
                        if lower==config.LOWER_BOUND_DAYS and route=='c':
                            x,_=training_user_percentiles(train,train,score)
                            _,head=common_head(x[:,None],y,x[:1,None])
                            coefficient=float(head.named_steps['logisticregression'].coef_[0,0])
                            coefficients.append({'model':model,'protocol':'percentile_head','task':task,'route':'C','lower_days':lower,
                                'feature':'train_user_score_percentile','is_model_score':True,'scaled_coefficient':coefficient,
                                'sign':'positive' if coefficient>0 else 'negative' if coefficient<0 else 'zero',
                                'train_pairs':len(train),'representation_end_day':config.HEAD_REPRESENTATION_END_DAY})
                            for split in ('validation','test'):
                                evaluation=cohort.filter(pl.col('split')==split)
                                values,fallback=training_user_percentiles(train,evaluation,score)
                                probability=head.predict_proba(values[:,None])[:,1]
                                result,auc=evaluate_probabilities(evaluation['uid'],evaluation['label_c'],probability)
                                percentile_rows.append({'model':model,'protocol':'percentile_head','task':task,'route':'C','split':split,
                                    'lower_days':lower,'is_main_lower':True,'fallback_pairs':fallback,**result})
                                _save_auc(config.PROCESSED/'percentile_diagnostics',model,'percentile_head',task,'c',split,lower,auc)
            print(f'raw score and head signs completed: {model}',flush=True)
    for rows,filename in [(raw_rows,outputs[0]),(diagnostic_rows,outputs[1]),(coefficients,outputs[2]),(percentile_rows,outputs[3])]:
        pl.DataFrame(rows).write_csv(config.RESULTS/filename)
    (config.RESULTS/'score_protocol_completed.json').write_text(json.dumps({'version':2,'primary':'raw_score',
        'models':[name for _,models in SCORE_SPECS for name in models],'diagnostic':['Recent7dPlays'],
        'head_refit':'main C test probabilities asserted equal to existing secondary predictions',
        'percentile':'train-only per-user midrank empirical CDF; pooled train fallback; L1 C only'})+'\n')
    return pl.DataFrame(raw_rows)


def write_score_protocol_report(results):
    from .metrics import bootstrap_mean
    percentile=pl.read_csv(config.RESULTS/'percentile_head_metrics.csv').with_columns(
        pl.col('model').str.replace(r'_seed\d+$','').alias('family'))
    keys=['family','protocol','task','route','split','lower_days','is_main_lower']
    rows=[]
    for group in percentile.partition_by(keys):
        row={key:group[key][0] for key in keys}
        auc=pl.concat([pl.read_parquet(config.PROCESSED/'percentile_diagnostics/user_auc'/
            f"{model}_percentile_head_{row['task']}_c_{row['split']}_l{row['lower_days']}.parquet").select('uid','auc') for model in group['model']])
        averaged=auc.group_by('uid').agg(pl.col('auc').mean()).sort('uid')
        low,high=bootstrap_mean(averaged['auc'])
        rows.append({**row,'user_auc':averaged['auc'].mean(),'user_auc_ci_low':low,'user_auc_ci_high':high,
            'seeds':group.height,'fallback_pairs_per_seed':group['fallback_pairs'].mean()})
    auxiliary=pl.DataFrame(rows)
    auxiliary.write_csv(config.RESULTS/'percentile_head_summary.csv')
    recent=pl.concat([pl.read_csv(config.RESULTS/'recent7d_raw_metrics.csv'),pl.read_csv(config.RESULTS/'recent7d_head_metrics.csv')],how='diagonal_relaxed').filter(pl.col('model')=='Recent7dPlays')
    recent.write_csv(config.RESULTS/'recent7d_diagnostic_metrics.csv')
    signs=pl.read_csv(config.RESULTS/'model_head_coefficients.csv')
    lines=['# 生スコア主評価と補助プロトコル','','主モデルの学習・ACT-Rの定義は維持。主指標は設計上の向きの生スコアAUC。ヘッドは副指標。全Lの明細はall_model_results.csv、全係数はmodel_head_coefficients.csv。SANSAは資源制約により未測定。','','## 主指標とヘッド・係数の向き','','L=1/C/テスト。iALS/BPRはユーザー内で3シードのAUCを平均。係数は標準化後、学習だけで推定。数値の符号を各シードごとに表示する。','','| タスク | モデル | 生スコアAUC | ヘッドAUC | スコア係数（シード別） |','|---|---|---:|---:|---|']
    for task in ('T2','T1'):
        main=results.filter((pl.col('task')==task)&(pl.col('route')=='C')&(pl.col('split')=='test')&(pl.col('lower_days')==config.LOWER_BOUND_DAYS)&(pl.col('protocol')=='raw_score')).sort('user_auc',descending=True)
        for row in main.iter_rows(named=True):
            head=results.filter((pl.col('family')==row['family'])&(pl.col('task')==task)&(pl.col('route')=='C')&(pl.col('split')=='test')&(pl.col('lower_days')==config.LOWER_BOUND_DAYS)&(pl.col('protocol')=='score_only')).row(0,named=True)
            c=signs.filter(pl.col('is_model_score')&(pl.col('protocol')=='score_only')&(pl.col('task')==task)&(pl.col('route')=='C')&(pl.col('lower_days')==config.LOWER_BOUND_DAYS)&(pl.col('model').str.replace(r'_seed\d+$','')==row['family'])).sort('model')
            coefficient=', '.join(f"{r['model']}: {r['scaled_coefficient']:+.6f}" for r in c.iter_rows(named=True))
            lines.append(f"| {task} | {row['family']} | {row['user_auc']:.6f} | {head['user_auc']:.6f} | {coefficient} |")
    lines += ['', '## 直前7日間の再生数（診断用）','','ユーザーの全経路・全再生率のイベント数。窓は[anchor−604800,anchor)。起点と同時刻の再生を全て除外。多いほど高い向きで固定し、主9種や上位モデル選定に混ぜない。','','| タスク | L | 生カウントAUC | ヘッドAUC | 基礎率＋カウントAUC |','|---|---:|---:|---:|---:|']
    for task in ('T2','T1'):
        for lower in config.SENSITIVITY_LOWER_DAYS:
            sub=recent.filter((pl.col('task')==task)&(pl.col('lower_days')==lower)&(pl.col('route')=='C')&(pl.col('split')=='test'))
            values={r['protocol']:r['user_auc'] for r in sub.iter_rows(named=True)}
            lines.append(f"| {task} | {lower} | {values['raw_score']:.6f} | {values['score_only']:.6f} | {values['baseline_plus_score']:.6f} |")
    lines += ['', '## 学習ユーザーCDFのパーセンタイル＋ヘッド（補助）','','ユーザーごとの学習スコアのみを参照するmidrank経験CDF。評価分布でfitしない。学習ユーザーがいなければ学習全体CDFを使う。学習29日／評価149日のスコア分布差とCDFの同点化が残る。L=1/Cに固定した追加診断で、主順位には使わない。','','| タスク | モデル | ユーザー内AUC | 95%CI | fallbackペア数/seed |','|---|---|---:|---|---:|']
    for r in auxiliary.filter(pl.col('split')=='test').sort('task','family').iter_rows(named=True):
        lines.append(f"| {r['task']} | {r['family']} | {r['user_auc']:.6f} | {r['user_auc_ci_low']:.6f}〜{r['user_auc_ci_high']:.6f} | {r['fallback_pairs_per_seed']:.0f} |")
    lines += ['', 'ヘッドの負係数による反転と、変換・有限精度による同点化を区別する。生スコアにlog-loss/Brierは定義しない。対応付きの全モデル差はpaired_model_differences.csvのraw_scoreが主、score_only/baseline_plus_scoreが副。', '']
    (config.RESULTS/'SCORE_PROTOCOL_REPORT.md').write_text('\n'.join(lines))
    return auxiliary
