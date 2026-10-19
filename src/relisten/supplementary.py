"""主結果を変更しない既知曲コホートとA/B/Cの順位比較。"""
import json

import numpy as np
import polars as pl
from scipy.stats import kendalltau

from . import config
from .analysis import family_name
from .labels import with_labels
from .light_models import benchmark_pairs
from .metrics import bootstrap_mean,user_auc_table,paired_user_auc_difference
from .score_protocol import SCORE_SPECS


def observed_items(events,end_day):
    return events.filter(pl.col('timestamp')<(end_day+1)*config.SECONDS_PER_DAY).select('item_id').unique().sort('item_id')


def seen_pair_flags(pairs,vocabularies):
    rows=[]
    common=vocabularies[config.HEAD_REPRESENTATION_END_DAY].filter(vocabularies[config.HEAD_REPRESENTATION_END_DAY].is_in(vocabularies[config.REPRESENTATION_END_DAY].implode()))
    for split,day in [('train',config.HEAD_REPRESENTATION_END_DAY),('validation',config.REPRESENTATION_END_DAY),('test',config.REPRESENTATION_END_DAY)]:
        group=pairs.filter(pl.col('split')==split)
        if group.is_empty():continue
        assert group['anchor_timestamp'].min()>=(day+1)*config.SECONDS_PER_DAY
        rows.append(group.select('pair_id').with_columns(pl.lit(day).alias('model_end_day'),
            pl.Series('seen_in_training',group['item_id'].is_in(vocabularies[day].implode())),
            pl.Series('seen_in_both_training',group['item_id'].is_in(common.implode()))))
    return pl.concat(rows).sort('pair_id')


def average_seed_auc(tables):
    reference=tables[0].select('uid','n','positives').sort('uid')
    for table in tables[1:]:
        if not reference.equals(table.select('uid','n','positives').sort('uid')):raise ValueError('Seed cohorts differ')
    return reference.join(pl.concat([t.select('uid','auc') for t in tables]).group_by('uid').agg(pl.col('auc').mean()),on='uid').sort('uid')


def subset_statistics(frame,column,fallback_column=None):
    restricted=frame.filter(pl.col('seen_in_training'))
    if not len(restricted):raise ValueError('No seen-item pairs')
    auc=user_auc_table(restricted['uid'],restricted['label_c'],restricted[column].to_numpy())
    minimum=float(frame[column].min())
    result={'pairs_before':len(frame),'pairs_after':len(restricted),'users_before':frame['uid'].n_unique(),
        'users_after':restricted['uid'].n_unique(),'minimum_score':minimum,
        'minimum_fraction_before':float((frame[column]==minimum).mean()),
        'minimum_fraction_after':float((restricted[column]==minimum).mean()),
        'fallback_fraction_before':float(frame[fallback_column].mean()) if fallback_column else None,
        'fallback_fraction_after':float(restricted[fallback_column].mean()) if fallback_column else None}
    return result,auc


def run_seen_item_analysis():
    pairs=benchmark_pairs()
    events=pl.scan_parquet(config.RAW/'flat/50m/listens.parquet').select('item_id',pl.col('timestamp').cast(pl.Int64))
    vocabularies={day:observed_items(events,day).collect()['item_id']
                  for day in (config.HEAD_REPRESENTATION_END_DAY,config.REPRESENTATION_END_DAY)}
    flags=seen_pair_flags(pairs,vocabularies)
    root=config.PROCESSED/'seen_item_diagnostics';root.mkdir(exist_ok=True)
    flags.write_parquet(root/'pair_flags.parquet')
    for day,items in vocabularies.items():items.to_frame('item_id').write_parquet(root/f'observed_items_day_{day}.parquet')
    rows=[];seed_rows=[]
    for filename,models in SCORE_SPECS:
        features=pl.read_parquet(config.PROCESSED/filename)
        columns=[c for c in features.columns if c not in pairs.columns or c=='pair_id']
        frame=pairs.join(flags,on='pair_id').join(features.select([c for c in columns if c!='model_end_day']),on='pair_id')
        assert frame.height==pairs.height and frame['pair_id'].n_unique()==pairs.height
        families=sorted(set(family_name(m) for m in models))
        for lower in config.SENSITIVITY_LOWER_DAYS:
            labeled=with_labels(frame,lower)
            for task in ('T2','T1'):
                for split in ('train','validation','test'):
                    cohort=labeled.filter((pl.col('task')==task)&(pl.col('split')==split))
                    day=config.HEAD_REPRESENTATION_END_DAY if split=='train' else config.REPRESENTATION_END_DAY
                    base_cohort=cohort
                    for selection,flag in [('matching_snapshot','seen_in_training'),('seen_by_both','seen_in_both_training')]:
                        cohort=base_cohort.with_columns(pl.col(flag).alias('seen_in_training'))
                        for family in families:
                            full_tables=[];seen_tables=[];statistics=[]
                            for model,column in models.items():
                                if family_name(model)!=family:continue
                                column=f'{column}_l{lower}' if f'{column}_l{lower}' in features.columns else column
                                path=config.PROCESSED/'raw_scores/user_auc'/f'{model}_raw_score_{task}_c_{split}_l{lower}.parquet'
                                # 元評価はvalidation/testのみ保存。trainは同じ生スコアから記述統計として計算する。
                                full=pl.read_parquet(path) if path.exists() else user_auc_table(cohort['uid'],cohort['label_c'],cohort[column].to_numpy())
                                fallback=f'{family}_cold' if f'{family}_cold' in cohort.columns else 'audio_missing' if family=='AudioCosine' else None
                                stats,seen=subset_statistics(cohort,column,fallback)
                                statistics.append(stats);full_tables.append(full);seen_tables.append(seen)
                                seed_rows.append({'model':model,'family':family,'task':task,'split':split,'lower_days':lower,
                                    'model_end_day':day,'selection':selection,**stats,'user_auc_before':full['auc'].mean(),'user_auc_after':seen['auc'].mean()})
                            before,after=average_seed_auc(full_tables),average_seed_auc(seen_tables)
                            low0,high0=bootstrap_mean(before['auc']);low1,high1=bootstrap_mean(after['auc'])
                            paired=paired_user_auc_difference(after,before)
                            row={'family':family,'task':task,'split':split,'route':'C','protocol':'raw_score','lower_days':lower,
                                'model_end_day':day,'selection':selection,'seeds':len(statistics),**{k:v for k,v in statistics[0].items() if k!='minimum_score'},
                                'auc_users_before':len(before),'auc_users_after':len(after),'user_auc_before':before['auc'].mean(),
                                'before_ci_low':low0,'before_ci_high':high0,'user_auc_after':after['auc'].mean(),
                                'after_ci_low':low1,'after_ci_high':high1,'auc_change':after['auc'].mean()-before['auc'].mean(),
                                'common_auc_users':paired['paired_users'],'common_user_auc_change':paired['mean_auc_difference'],
                                'common_change_ci_low':paired['difference_ci_low'],'common_change_ci_high':paired['difference_ci_high']}
                            for key in ['minimum_fraction_before','minimum_fraction_after','fallback_fraction_before','fallback_fraction_after']:
                                values=[s[key] for s in statistics if s[key] is not None]
                                row[key]=float(np.mean(values)) if values else None
                            rows.append(row)
        print(f'seen-item auxiliary evaluation completed: {filename}',flush=True)
    table=pl.from_dicts(rows,infer_schema_length=None).sort('lower_days','task','split','family')
    table.write_csv(config.RESULTS/'seen_item_auc_comparison.csv')
    pl.from_dicts(seed_rows,infer_schema_length=None).write_csv(config.RESULTS/'seen_item_auc_by_seed.csv')
    (config.RESULTS/'seen_item_protocol.json').write_text(json.dumps({
        'auxiliary_only':True,'main_results_modified':False,'route':'C','lower_days':list(config.SENSITIVITY_LOWER_DAYS),
        'selection':'same global observed-item set for every model; all routes and played ratios, not model-specific eligibility',
        'selections':{'matching_snapshot':'train29/evaluation149 catalog','seen_by_both':'intersection of29 and149 catalogs; scores still use original29/149 model assignment'},
        'train_cutoff_day':config.HEAD_REPRESENTATION_END_DAY,'evaluation_cutoff_day':config.REPRESENTATION_END_DAY,
        'cutoff':'timestamp < (end_day+1)*86400; never use149-day vocab for train pairs',
        'vocabulary_sizes':{str(d):len(v) for d,v in vocabularies.items()},
        'minimum_fraction':'score exactly equals minimum in unfiltered same task/split/L/model/seed; threshold fixed after filtering',
        'fallback_fraction':'stored CF/SASRec cold flags or AudioCosine missing flag; null for models without explicit fallback',
        'seed_aggregation':'per-user mean AUC over3 seeds, then1000-user bootstrap; minimum/fallback fractions averaged over seeds',
        'auc_change':'restricted macro AUC minus original macro AUC; eligible user population may change',
        'common_user_change':'separate paired user bootstrap on users eligible both before and after',
        'train_rows':'descriptive training-cohort raw AUC, not the held-out main result',
    },indent=2)+'\n')
    return table


def combined_rank_table(results,task_a):
    data=results.filter((pl.col('protocol')=='raw_score')&(pl.col('lower_days')==config.LOWER_BOUND_DAYS)
        &(pl.col('split')=='test')&pl.col('route').is_in(['B','C'])&pl.col('family').is_in(config.BENCHMARK_MODELS))
    a=task_a.group_by('model').agg(pl.col('ndcg_at_10').mean().alias('a_ndcg_at_10')).rename({'model':'family'})
    expected=set(data['family'])
    if set(a['family'])!=expected:raise ValueError('A/B/C model sets differ')
    tables=[]
    for task in ('T2','T1'):
        bc=data.filter(pl.col('task')==task).select('family','route','user_auc').pivot(on='route',index='family',values='user_auc').rename({'B':'b_user_auc','C':'c_user_auc'})
        if set(bc['family'])!=expected:raise ValueError('T1/T2 model sets differ')
        if bc.select(pl.any_horizontal(pl.all().is_null()).any()).item():raise ValueError('Missing B/C model scores')
        table=bc.join(a,on='family',how='left')
        if table['a_ndcg_at_10'].null_count():raise ValueError('Task A model missing')
        ac=kendalltau(table['a_ndcg_at_10'],table['c_user_auc'],variant='b')
        bc_tau=kendalltau(table['b_user_auc'],table['c_user_auc'],variant='b')
        table=table.with_columns(pl.lit(task).alias('task'),pl.lit(config.LOWER_BOUND_DAYS).alias('lower_days'),
            pl.lit(len(table)).alias('rank_models'),pl.lit(float(ac.statistic)).alias('tau_a_c'),pl.lit(float(ac.pvalue)).alias('p_a_c'),
            pl.lit(float(bc_tau.statistic)).alias('tau_b_c'),pl.lit(float(bc_tau.pvalue)).alias('p_b_c'),
            pl.col('a_ndcg_at_10').rank(descending=True,method='average').alias('a_rank'),
            pl.col('b_user_auc').rank(descending=True,method='average').alias('b_rank'),
            pl.col('c_user_auc').rank(descending=True,method='average').alias('c_rank'))
        tables.append(table)
    return pl.concat(tables).sort('task','c_rank')


def write_supplementary_report(seen,ranks):
    ranks.write_csv(config.RESULTS/'abc_rank_summary.csv')
    labels={'matching_snapshot':'期限対応','seen_by_both':'29∩149'}
    def percentage(value):return '<0.01%' if 0<value<.0001 else f'{100*value:.2f}%'
    lines=['# 既知曲限定の補助評価とA/B/C順位比較','','主モデル・主結果の変更なし。生スコアは設計上高いほど合う向きで固定。SANSAは資源制約で未測定。','','## 学習期間の出現曲に限定（L=1/C/テスト）','','学習用ペアは0〜29日、検証・テストは0〜149日の元ログに出現した曲に限定。モデルごとの再生率フィルタをかける前の共通集合。期限対応の集合に加え、29日版と149日版の両方に出現した共通曲集合（29∩149）も併記する。どちらも保存済みスコアのモデル期限は学習29日/評価149日のまま。全分割とL=3/7/14はCSVに保存。95%区間は1000回ユーザーブートストラップ。iALS/BPRはユーザー内で先に3シード平均。','','| 曲集合 | タスク | モデル | 元AUC [95%区間] | 限定後AUC [95%区間] | 差 | 最低値率 前→後 | 欠損最低値率 前→後 |','|---|---|---|---|---|---:|---|---|']
    for selection in ('matching_snapshot','seen_by_both'):
        for task in ('T2','T1'):
            main=seen.filter((pl.col('task')==task)&(pl.col('split')=='test')&(pl.col('lower_days')==config.LOWER_BOUND_DAYS)&(pl.col('selection')==selection)).sort('user_auc_after',descending=True)
            for r in main.iter_rows(named=True):
                fallback='対象外' if r['fallback_fraction_before'] is None else f"{percentage(r['fallback_fraction_before'])}→{percentage(r['fallback_fraction_after'])}"
                lines.append(f"| {labels[selection]} | {task} | {r['family']} | {r['user_auc_before']:.6f} [{r['before_ci_low']:.6f}, {r['before_ci_high']:.6f}] | {r['user_auc_after']:.6f} [{r['after_ci_low']:.6f}, {r['after_ci_high']:.6f}] | {r['auc_change']:+.6f} | {percentage(r['minimum_fraction_before'])}→{percentage(r['minimum_fraction_after'])} | {fallback} |")
    lines += ['', '対象ペアとユーザー数：','']
    for selection in ('matching_snapshot','seen_by_both'):
        for task in ('T2','T1'):
            r=seen.filter((pl.col('task')==task)&(pl.col('split')=='test')&(pl.col('lower_days')==config.LOWER_BOUND_DAYS)&(pl.col('selection')==selection)).row(0,named=True)
            lines.append(f"- {labels[selection]} / {task}: {r['pairs_before']:,}→{r['pairs_after']:,}ペア、AUC対象{r['auc_users_before']:,}→{r['auc_users_after']:,}ユーザー。")
    lines += ['', '出現曲で絞っても、未学習ユーザーやモデル内部の再生率条件などによる採用外は残りうる。最低値は絞り込み前の同じタスク・分割・L・モデル・シード内の観測最小値に固定。ACT-Rの最低値率は観測最小値で、欠損の割合ではない。明示的な欠損フラグがないモデルの欠損最低値率は対象外。差は集団を限定した前後差であり、ユーザー構成も変わりうる。同じユーザーだけの対応付き差と人数はseen_item_auc_comparison.csvに別記。', '', '## A/C・B/CのKendall τとB/Cの生スコアAUC（L=1・テスト）', '', 'Aは既存149日モデルによる150〜210日の複数正解holdout NDCG@10。同じAをT1/T2それぞれのB/Cモデル順位と比較する。T1専用・T2専用の次曲タスクを新設したものではない。τ-bはモデルを観測単位として計算し、シードはモデル内で平均。モデル数8、p値は多重比較未補正。', '', '| タスク | モデル | A NDCG@10 | B AUC | C AUC | A順位 | B順位 | C順位 | τ(A,C) | τ(B,C) |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ranks.sort(['task','c_rank'],descending=[True,False]).iter_rows(named=True):
        lines.append(f"| {r['task']} | {r['family']} | {r['a_ndcg_at_10']:.6f} | {r['b_user_auc']:.6f} | {r['c_user_auc']:.6f} | {r['a_rank']:g} | {r['b_rank']:g} | {r['c_rank']:g} | {r['tau_a_c']:.6f} | {r['tau_b_c']:.6f} |")
    lines += ['', '出力：seen_item_auc_comparison.csv（集約）、seen_item_auc_by_seed.csv（各シード）、seen_item_protocol.json（定義）、abc_rank_summary.csv（統合表、τのp値を含む）。', '']
    (config.RESULTS/'SUPPLEMENTARY_COMPARISON_REPORT.md').write_text('\n'.join(lines))
