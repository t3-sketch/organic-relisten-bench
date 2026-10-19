"""既知曲CとAの順位相関、および同一ペア・ユーザーでのB−C差。"""
import json

import numpy as np
import polars as pl
from scipy.stats import kendalltau

from . import config
from .analysis import family_name
from .labels import with_labels
from .light_models import benchmark_pairs
from .metrics import user_auc_table,paired_user_auc_difference
from .score_protocol import SCORE_SPECS
from .supplementary import average_seed_auc


def paired_route_auc(frame,columns):
    if frame['pair_id'].n_unique()!=len(frame):raise ValueError('Duplicate paired-route pair IDs')
    tables={route:[] for route in ('b','c')}
    for column in columns:
        for route in tables:
            tables[route].append(user_auc_table(frame['uid'],frame[f'label_{route}'],frame[column].to_numpy()))
    b,c=(average_seed_auc(tables[route]) for route in ('b','c'))
    common=b.rename({'auc':'b_auc','n':'b_n','positives':'b_positives'}).join(
        c.rename({'auc':'c_auc','n':'c_n','positives':'c_positives'}),on='uid',how='inner',validate='1:1').sort('uid')
    if not len(common):raise ValueError('No users with both B and C AUC defined')
    assert common['b_n'].equals(common['c_n']),'B/C pair counts differ'
    paired=paired_user_auc_difference(common.select('uid',pl.col('b_auc').alias('auc')),
        common.select('uid',pl.col('c_auc').alias('auc')),n_bootstrap=1000)
    common=common.with_columns((pl.col('b_auc')-pl.col('c_auc')).alias('b_minus_c'))
    return {'pairs':len(frame),'users':frame['uid'].n_unique(),'b_auc_users':len(b),'c_auc_users':len(c),
        'paired_pairs':int(common['b_n'].sum()),'b_auc_common_users':common['b_auc'].mean(),
        'c_auc_common_users':common['c_auc'].mean(),'seeds':len(columns),**paired},common


def seen_a_c_ranks(seen,task_a):
    data=seen.filter((pl.col('selection')=='matching_snapshot')&(pl.col('split')=='test')
        &(pl.col('lower_days')==config.LOWER_BOUND_DAYS))
    a=task_a.group_by('model').agg(pl.col('ndcg_at_10').mean().alias('a_ndcg_at_10')).rename({'model':'family'})
    summaries=[];details=[]
    for task in ('T2','T1'):
        c=data.filter(pl.col('task')==task).select('family','user_auc_after','pairs_after','auc_users_after')
        if set(c['family'])!=set(a['family']) or c['family'].n_unique()!=len(c):raise ValueError('Seen C/A model sets differ')
        table=c.join(a,on='family',validate='1:1')
        tau=kendalltau(table['a_ndcg_at_10'],table['user_auc_after'],variant='b')
        summary={'task':task,'lower_days':config.LOWER_BOUND_DAYS,'selection':'matching_snapshot','split':'test',
            'models':len(table),'kendall_tau':float(tau.statistic),'p_value':float(tau.pvalue),
            'c_pairs':int(table['pairs_after'][0]),'c_auc_users':int(table['auc_users_after'][0])}
        summaries.append(summary)
        details.append(table.with_columns(pl.lit(task).alias('task'),
            pl.col('a_ndcg_at_10').rank(descending=True,method='average').alias('a_rank'),
            pl.col('user_auc_after').rank(descending=True,method='average').alias('seen_c_rank')))
    return pl.DataFrame(summaries),pl.concat(details).sort('task','seen_c_rank')


def run_paired_route_supplement():
    tau,detail=seen_a_c_ranks(pl.read_csv(config.RESULTS/'seen_item_auc_comparison.csv'),
        pl.read_csv(config.RESULTS/'task_a_next_item.csv'))
    pairs=with_labels(benchmark_pairs(),config.LOWER_BOUND_DAYS).filter(pl.col('split')=='test')
    flags=pl.read_parquet(config.PROCESSED/'seen_item_diagnostics/pair_flags.parquet').select('pair_id','seen_in_training','model_end_day')
    pairs=pairs.join(flags,on='pair_id',validate='1:1')
    assert pairs.select((pl.col('model_end_day')==config.REPRESENTATION_END_DAY).all()).item()
    root=config.PROCESSED/'paired_route_diagnostics';root.mkdir(exist_ok=True)
    references={};rows=[]
    for filename,models in SCORE_SPECS:
        features=pl.read_parquet(config.PROCESSED/filename)
        score_columns={m:f'{col}_l{config.LOWER_BOUND_DAYS}' if f'{col}_l{config.LOWER_BOUND_DAYS}' in features.columns else col for m,col in models.items()}
        frame=pairs.join(features.select('pair_id',*score_columns.values()),on='pair_id',validate='1:1')
        assert frame.height==pairs.height,'Scores missing from B/C cohort'
        for task in ('T2','T1'):
            full=frame.filter(pl.col('task')==task)
            for cohort in ('all_pairs','matching_snapshot'):
                sub=full if cohort=='all_pairs' else full.filter(pl.col('seen_in_training'))
                for family in sorted(set(family_name(m) for m in models)):
                    columns=[col for model,col in score_columns.items() if family_name(model)==family]
                    summary,users=paired_route_auc(sub,columns)
                    identity=users.select('uid','b_n','c_n','b_positives','c_positives')
                    key=(task,cohort)
                    if key in references:assert references[key].equals(identity),'B/C users differ across models'
                    else:references[key]=identity
                    users.write_parquet(root/f'{family}_{task}_{cohort}_l{config.LOWER_BOUND_DAYS}.parquet')
                    rows.append({'family':family,'task':task,'cohort':cohort,'lower_days':config.LOWER_BOUND_DAYS,
                        'split':'test','protocol':'raw_score',**summary})
        print(f'paired B-C evaluation completed: {filename}',flush=True)
    differences=pl.DataFrame(rows).sort('task','cohort','family')
    tau.write_csv(config.RESULTS/'seen_item_a_c_kendall.csv')
    detail.write_csv(config.RESULTS/'seen_item_a_c_rank_detail.csv')
    differences.write_csv(config.RESULTS/'paired_b_minus_c.csv')
    (config.RESULTS/'paired_route_protocol.json').write_text(json.dumps({
        'auxiliary_only':True,'main_results_modified':False,'lower_days':config.LOWER_BOUND_DAYS,'split':'test',
        'raw_direction':'original higher-is-better; identical score column for B and C',
        'cohorts':['all_pairs','matching_snapshot'],'seen_cohort':'original shared raw catalog; matching29/149 deadline, test149',
        'auc_unit':'per-user AUC over identical pairs, separately labeled B and C',
        'paired_users':'intersection of users with positive and negative labels for both B and C; all their cohort pairs retained',
        'seed_aggregation':'mean per-user AUC over3 seeds separately for B/C, then subtract, then user bootstrap',
        'difference':'B minus C; positive favors B; no separate-user-population macro subtraction',
        'bootstrap_replicates':1000,'seed':config.SEED,'confidence_interval':'95% percentile, unadjusted',
        'rank_test':'Kendall tau-b on8 model families; A seed mean NDCG@10 versus known-item C mean user AUC',
        'A_unchanged':'existing149-day representation and150-210 multi-positive holdout; A catalog/targets not re-filtered',
        'rank_p_value':'two-sided scipy kendalltau, method auto; unadjusted across comparisons',
    },indent=2)+'\n')
    write_paired_route_report(tau,differences)
    return tau,differences


def write_paired_route_report(tau,differences):
    lines=['# 期限対応の既知曲CとA・対応付きB−C補助比較','','主結果・モデル・スコアを変更しないL=1・テストの補助分析。SANSAは資源制約で未測定。','','## 期限対応の既知曲CとAの順位相関','','Cは既存の全経路・全再生率の出現曲集合（評価149日）に限定したユーザー内AUC。Aは既存のNDCG@10をそのまま使う。モデル内でシード平均後、8モデルを単位としてKendall τ-bを計算。p値は両側・多重比較未補正。','','| タスク | モデル数 | 既知曲Cのペア数 | CのAUC対象ユーザー | τ(A,既知曲C) | p値 |','|---|---:|---:|---:|---:|---:|']
    for r in tau.iter_rows(named=True):
        lines.append(f"| {r['task']} | {r['models']} | {r['c_pairs']:,} | {r['c_auc_users']:,} | {r['kendall_tau']:.6f} | {r['p_value']:.6f} |")
    lines += ['', 'モデル別のA/C指標と順位はseen_item_a_c_rank_detail.csv。Aを既知曲限定で再学習・再評価した比較ではない。','','## 同じペア・同じユーザーでのB−C差','','各コホート内の同じスコア・同じペアへB/Cラベルを付け、両方のAUCが定義できる共通ユーザーに限定する。各ユーザーのB AUCとC AUC（iALS/BPRは各3シード平均）の差を1000回、ユーザー単位で再標本化。区間は95%パーセンタイル、多重比較未補正。正の差はBが高いことを表す。','','| タスク | 曲集合 | 対応ペア数 | 対応ユーザー数 | 元のB AUC対象数 | 元のC AUC対象数 |','|---|---|---:|---:|---:|---:|']
    for task in ('T2','T1'):
        for cohort in ('all_pairs','matching_snapshot'):
            r=differences.filter((pl.col('task')==task)&(pl.col('cohort')==cohort)).row(0,named=True)
            lines.append(f"| {task} | {'全ペア' if cohort=='all_pairs' else '期限対応の既知曲'} | {r['paired_pairs']:,} | {r['paired_users']:,} | {r['b_auc_users']:,} | {r['c_auc_users']:,} |")
    lines += ['', '| モデル | T2 全ペア B−C [95%区間] | T2 既知曲 B−C [95%区間] | T1 全ペア B−C [95%区間] | T1 既知曲 B−C [95%区間] |','|---|---|---|---|---|']
    for family in sorted(differences['family'].unique()):
        cells=[]
        for task,cohort in [('T2','all_pairs'),('T2','matching_snapshot'),('T1','all_pairs'),('T1','matching_snapshot')]:
            r=differences.filter((pl.col('family')==family)&(pl.col('task')==task)&(pl.col('cohort')==cohort)).row(0,named=True)
            cells.append(f"{r['mean_auc_difference']:+.6f} [{r['difference_ci_low']:+.6f}, {r['difference_ci_high']:+.6f}]")
        lines.append('| '+family+' | '+' | '.join(cells)+' |')
    lines += ['', '既存のB/C表はそれぞれAUC対象ユーザーが異なるため、そちらの集約値を単純に引いた差とは一致しない場合がある。対応後のB/C平均と人数はpaired_b_minus_c.csvに保存。ユーザー別の両AUC・差・正例数・ペア数はdata/processed/paired_route_diagnostics/。定義はpaired_route_protocol.json。', '']
    (config.RESULTS/'PAIRED_ROUTE_SUPPLEMENT_REPORT.md').write_text('\n'.join(lines))
