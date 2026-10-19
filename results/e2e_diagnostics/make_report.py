"""Render the frozen inference diagnostics; no training or model changes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    loss = pl.read_csv(OUT / 'loss_by_epoch.csv')
    metrics = pl.read_csv(OUT / 'metrics.csv')
    checks = pl.read_csv(OUT / 'cohort_checks.csv')
    dist = pl.read_csv(OUT / 'score_distributions.csv')
    verification = json.loads((OUT / 'verification.json').read_text())
    lower_values = sorted(loss['lower_days'].unique().to_list())

    def metric(lower, split, subset='all', kind='score'):
        return metrics.filter((pl.col('lower_days') == lower) & (pl.col('split') == split)
                              & (pl.col('subset') == subset) & (pl.col('score_kind') == kind)).row(0, named=True)

    def check(lower, split):
        return checks.filter((pl.col('lower_days') == lower) & (pl.col('split') == split)).row(0, named=True)

    def distribution(lower, kind='score'):
        return dist.filter((pl.col('lower_days') == lower) & (pl.col('split') == 'test')
                           & (pl.col('subset') == 'known') & (pl.col('kind') == kind)).row(0, named=True)

    def auc(m):
        return f"{m['user_auc']:.6f} [{m['user_auc_ci_low']:.6f}, {m['user_auc_ci_high']:.6f}]"

    fig, ax = plt.subplots(figsize=(7, 4))
    for lower in lower_values:
        f = loss.filter(pl.col('lower_days') == lower)
        ax.plot(f['epoch'], f['loss'], marker='o', label=f'L={lower}')
    ax.set(xlabel='Epoch', ylabel='Online training BCE', xticks=range(1, 11))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / 'training_loss.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for lower, ax in zip(lower_values, axes.flat):
        f = pl.read_parquet(OUT / f'scores_test_l{lower}.parquet').filter(pl.col('known'))
        ax.hist(f['score'], bins=np.linspace(-1, 1, 51), weights=np.full(len(f), 1 / len(f)))
        ax.set(title=f'L={lower}: known target only', xlabel='Score before scale/bias', ylabel='Fraction of known pairs')
    fig.tight_layout(); fig.savefig(OUT / 'test_known_score_histogram.png', dpi=160); plt.close(fig)

    loss_table = table(['epoch'] + [f'L={l}' for l in lower_values],
                       [[e] + [f"{loss.filter((pl.col('lower_days') == l) & (pl.col('epoch') == e))['loss'][0]:.6f}" for l in lower_values]
                        for e in range(1, 11)])
    fit_rows = []
    for l in lower_values:
        m = metric(l, 'train'); c = check(l, 'train')
        fit_rows.append([l, f"{m['pairs']:,}", f"{m['auc_users']:,}", auc(m),
                         f"{c['final_eval_bce']:.6f}", f"{c['floor_fraction'] * 100:.3f}%",
                         auc(metric(l, 'train', 'known'))])
    floor_rows, scatter_rows, compare_rows, probability_rows = [], [], [], []
    for l in lower_values:
        c = check(l, 'test'); d = distribution(l); p = distribution(l, 'probability')
        floor_probability = pl.read_parquet(OUT / f'scores_test_l{l}.parquet').filter(~pl.col('known'))['probability'][0]
        floor_rows.append([l, f"{c['pairs']:,}", f"{c['floor_pairs']:,}", f"{100*c['floor_fraction']:.3f}%",
                           f"{100*c['floor_positive_rate']:.3f}%", f"{100*floor_probability:.3f}%",
                           f"{100*c['known_positive_rate']:.3f}%"])
        scatter_rows.append([l, f"{d['n']:,}", f"{d['mean']:.6f}", f"{d['std']:.6f}",
                             f"{d['p01']:.6f}", f"{d['p25']:.6f}", f"{d['median']:.6f}",
                             f"{d['p75']:.6f}", f"{d['p99']:.6f}", f"{d['unique']:,}",
                             f"{100*c['known_score_minus_one_fraction']:.3f}%", f"{100*c['known_score_plus_one_fraction']:.6f}%"])
        compare_rows.append([l, auc(metric(l, 'train')), auc(metric(l, 'test')),
                             auc(metric(l, 'test', 'known')), auc(metric(l, 'test', kind='probability'))])
        probability_rows.append([l, f"{c['scale']:.6f}", f"{c['bias']:.6f}",
                                 f"{p['min']:.6f}", f"{p['median']:.6f}", f"{p['max']:.6f}", f"{p['std']:.6f}",
                                 f"{c['saved_probability_max_abs_difference']:.3g}",
                                 f"{100*c['saved_probability_exact_fraction']:.3f}%"])
    body = f'''# End-to-End固定重みの診断

## 実施範囲と定義

**Observed：** 最終epoch10の保存済みcheckpointを読み、`eval()`と`torch.inference_mode()`で学習ペア（起点30〜59日）とtest（165〜210日）を推論した。T2／C、L=1を主条件、L=3/7/14を感度条件とする。新しい学習、optimizer更新、設定変更、既存結果の上書きは行っていない。履歴は既存`PairDataset`と`_histories`を使い、起点より前の適格再生と、条件を満たす現在の推薦1回を入力する。曲集合は元実験と同じ29日版である。

内部スコアを` s = tanh(h・e)`、未知曲では`s = −1.000001`とし、`logit = softplus(log_scale) × s + bias`、`p = sigmoid(logit)`と区別する。最低値は**内部スコア**の値であり、logitや確率の値ではない。float32の実値は約−1.0000009536743164。未知曲はitem IDの集合所属でも独立に確認した。

user AUCは、正負両方を持つ各ユーザーのAUCを等重み平均し、同点に0.5を与える。区間はユーザー単位bootstrap1000回、seed={config_seed()}。学習ペアの区間は**学習済みモデルの同じ学習標本への当てはまり**の記述で、未見データへの汎化の区間ではない。高いほど合うという設計上の向きを固定し、反転はしていない。

## 1. 保存された学習損失の推移

{loss_table}

![保存ログの損失推移]({OUT / 'training_loss.png'})

**Observed：** すべてのLで保存損失はepochごとに低下した。損失は各batchの更新前BCEをペア数で重み付けしたepoch平均であり、epoch終了後の固定重みを全学習ペアへ適用したBCEとは異なる。epochごとのAUCは未保存で、当時の重みは上書きされているため、本診断は最終epochだけを評価する。

## 2. 最終モデル自身の学習ペアAUC

{table(['L','学習ペア','AUC対象user','内部スコアuser AUC [95%区間]','固定重みの学習BCE','学習の最低値割合','既知曲のみ学習AUC [95%区間]'], fit_rows)}

既知曲だけの列は診断用に母集団を限定して再集計したもので、全ペアの主結果を置き換えない。全ペアと対象ユーザーも変わりうる。

## 3. テストの最低値割合

{table(['L','testペア','最低値ペア','最低値割合','最低値群の正例率','最低値群の予測確率','既知曲群の正例率'], floor_rows)}

**Observed：** 最低値群は29日版曲集合に対象曲が存在しないペアであり、そのすべてに同じスコアが割り当てられる。これは既知曲のtanhが−1へ飽和したケースとは別である。最低値割合は学習の約10.5%からtestの約38.5〜38.8%へ増えている。

ここでの「未知曲」は**29日版の固定vocabularyにない曲**を意味する。30〜59日の教師ペアに登場した曲でも、この集合にない場合はtarget ID=0となり、独立した曲embeddingは追加されない。「教師データに登場しなかった曲」という意味ではない。

## 4. 最低値以外（既知曲）のテストスコア分布

{table(['L','ペア','平均','標準偏差','p01','p25','中央値','p75','p99','異なる値数','厳密に−1','厳密に＋1'], scatter_rows)}

標準偏差は母標準偏差（ddof=0）。分位点はペア単位の分布で、ユーザー等重みのAUCとは重みが異なる。CSVには最小・最大・p05・p95も保存した。±1の割合はfloat32での厳密一致である。

![最低値群を除いた内部スコア分布]({OUT / 'test_known_score_histogram.png'})

最終確率の分布と保存済みtest予測との一致も確認した。

{table(['L','正scale','bias','既知曲p最小','既知曲p中央値','既知曲p最大','既知曲p標準偏差','既存pとの最大絶対差','既存pとの厳密一致'], probability_rows)}

## 5. 原因の切り分けと限界

{table(['L','学習・全ペア内部AUC','test・全ペア内部AUC','test・既知曲内部AUC','test・全ペア確率AUC'], compare_rows)}

**Observed：** 上表で学習への当てはまり、testでの識別、既知曲に限った識別を分けて示した。内部スコアと確率は数学的には単調関係だが、浮動小数点で同点が増えればAUCに小差が出るため、両者を保存した。

**Inferred：** 学習AUCとtest AUCの乖離は汎化の問題と整合する。未知曲の増加は識別不能な同点と固定された低順位を増やす。ただし、未知曲の存在だけでtest性能の全低下を説明したとはいえない。既知曲だけのAUCは別母集団上の診断であり、全体の低下に対する因果的な寄与率ではない。

**Observed：L=1の具体的な切り分け。** 学習AUCは全ペア0.82325、既知曲0.92353。testは全ペア0.48068、既知曲0.53610である。既知曲スコアの標準偏差は0.44196、異なる値は501,249個、厳密な−1飽和は1.75%で、全件定数化ではない。未知曲群の観測正例率は10.72%だが、同群への予測確率は約2.32%である。

**Inferred：** この所見は、(a)未知曲へ割り当てる固定低順位・低確率とcoverage低下、(b)既知曲でも大きい学習・testの識別差、を別々の診断対象として支持する。(b)には学習への過適合、期間・ユーザー・曲の構成変化、表現年齢の影響等が含まれうるため、過適合だけを原因と確定しない。未知曲のスコアは定数に置き換わるので、その例からencoder・target embeddingへのスコア勾配は流れず、scale/biasの学習だけに寄与する（既存forwardの式からの推論）。

**Unverified：** epochごとの学習AUC、検証AUCによる最適epoch、未知曲対策やtanh変更の効果、複数seedでの再現性は調べていない。初期表現29日版と主比較149日版の違いも残る。今回の結果から設定・モデル・主結果を変更していない。

## 根拠・再現・保護

- 損失：元の`../sasrec_e2e_training_l{{1,3,7,14}}.csv`、集約`loss_by_epoch.csv`。
- 元モデルと入力：`../../data/processed/models/SASRec_e2e_l{{L}}.pt`、`day_29/SASRec_items.npy`、`../../data/processed/pairs.parquet`。
- 実装：[`PairDataset`](/Users/macuser/dev/paper1/src/relisten/end_to_end.py:23)、[`DirectRelistening.forward`](/Users/macuser/dev/paper1/src/relisten/end_to_end.py:53)、[学習損失集計](/Users/macuser/dev/paper1/src/relisten/end_to_end.py:101)、[user AUC](/Users/macuser/dev/paper1/src/relisten/metrics.py:9)。引用行は診断時点のもの。
- 新規診断：`scores_{{train,test}}_l{{L}}.parquet`、`metrics.csv`、`cohort_checks.csv`、`score_distributions.csv`、user単位AUC parquet。既存結果とは別フォルダ。
- 実行：repo rootで`.venv/bin/python -u results/e2e_diagnostics/diagnose.py`。その後`.venv/bin/python results/e2e_diagnostics/make_report.py`で文書・図を生成。
- device：{verification['device']}、推論・集計の実測時間：{verification['seconds']:.1f}秒。ピークメモリは未計測。
- `protected_before.json`と`verification.json`：既存{verification['protected_files']}ファイルを前後SHA256照合し一致。checkpoint、既存予測、設定、source、AGENTS.md・DECISIONS.md、既存結果を保護。今回のdiagnosticsディレクトリは対象外。
- 追加の学習・テストスイート実行はなし。全件推論のfiniteチェック、pair ID・uid・labelの一致、既存確率との最大差<1e−6、最終epoch10、全保護hash一致を実行時assertで検証した。
- 独立した指標照合：4条件×2分割×2母集団で各20ユーザー、計320件のuser AUCをscikit-learnと比較。最大絶対差は2.23e−16未満。詳細と環境versionは[verification.json]({OUT / 'verification.json'})。
- 集約データ：[学習・test指標]({OUT / 'metrics.csv'})、[スコア分布]({OUT / 'score_distributions.csv'})、[最低値と再現確認]({OUT / 'cohort_checks.csv'})、[全epoch損失]({OUT / 'loss_by_epoch.csv'})。
'''
    (OUT / 'REPORT.md').write_text(body)
    print(OUT / 'REPORT.md')


def config_seed():
    from relisten.config import SEED
    return SEED


if __name__ == '__main__':
    main()
