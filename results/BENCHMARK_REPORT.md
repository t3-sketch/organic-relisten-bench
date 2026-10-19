# Paper 1 実験結果

主設定：L=1、推薦再生から24時間以上〜90日後（両端を含む）の自発再聴。
TMがモデル結果を見る前に概念上の理由で決定。L=7・14は必須感度分析として保持。

## T2：未知曲の推薦後の自発再聴

主指標はヘッドを通さない生スコアのユーザー内AUC。全モデルの設計上の向き（高いほど合う）を固定し、符号は選び直さない。共通ヘッド経由のAUCは副指標として併記。ユーザーを等重みとし、正負両方を持つユーザーのみAUCを計算。

| モデル | 生スコアAUC（主） | 95%CI | ヘッドAUC（副） | ヘッド係数の符号 |
|---|---:|---|---:|---|
| AudioCosine | 0.5619 | 0.5570–0.5668 | 0.5619 | AudioCosine: +0.40035 |
| SASRec | 0.5317 | 0.5269–0.5366 | 0.5317 | SASRec: +0.01544 |
| DecayPop | 0.5195 | 0.5143–0.5244 | 0.5200 | DecayPop: +0.18275 |
| ACT-R | 0.5105 | 0.5064–0.5150 | 0.5105 | ACT-R: +0.03315 |
| MostPop | 0.5074 | 0.5022–0.5124 | 0.5074 | MostPop: +0.18102 |
| iALS | 0.5011 | 0.4964–0.5055 | 0.4989 | iALS_seed41: -0.11253, iALS_seed42: -0.11252, iALS_seed43: -0.11252 |
| ItemKNN | 0.5005 | 0.4957–0.5050 | 0.5005 | ItemKNN: +0.10822 |
| BPR | 0.4929 | 0.4884–0.4973 | 0.5071 | BPR_seed41: -0.10586, BPR_seed42: -0.10647, BPR_seed43: -0.10579 |

評価対象は967,884ペア、4,819ユーザー。AUCを計算できるのは4,124ユーザー。ユーザー基礎率だけのユーザー内AUCは0.5。

iALS/BPRは3シードの平均。信頼区間は各ユーザーのシード平均AUCを1000回のユーザーブートストラップで求める。
モデル間の対応付き差は `paired_model_differences.csv`。区間が0を含む差を優劣確定と扱わない。

## 下限の感度分析

| モデル | L=1 | L=3 | L=7 | L=14 |
|---|---:|---:|---:|---:|
| AudioCosine | 0.5619 | 0.5636 | 0.5649 | 0.5643 |
| SASRec | 0.5317 | 0.5321 | 0.5331 | 0.5353 |
| DecayPop | 0.5195 | 0.5197 | 0.5198 | 0.5220 |
| ACT-R | 0.5105 | 0.5095 | 0.5094 | 0.5106 |
| MostPop | 0.5074 | 0.5075 | 0.5075 | 0.5098 |
| iALS | 0.5011 | 0.5008 | 0.5009 | 0.5030 |
| ItemKNN | 0.5005 | 0.5005 | 0.5007 | 0.5040 |
| BPR | 0.4929 | 0.4925 | 0.4926 | 0.4945 |

各Lで同じ窓の活動30日以上条件を再計算。表は生スコアで、ヘッド副指標では対応する学習ラベルでヘッドを再学習。Lは結果を見て変更しない。

![下限の感度分析](figures/model_lower_bound_sensitivity.png)

## 上位モデル間の対応付き差

L=1・T2・C・生スコア（主指標）。各ユーザーのAUCの差を1000回、ユーザー単位で再標本化。iALS/BPRはユーザー内で先に3シードを平均する。

| 比較（左−右） | AUC差 | 95%CI | ユーザー数 |
|---|---:|---|---:|
| AudioCosine − SASRec | 0.030188 | 0.023298–0.037633 | 4124 |
| AudioCosine − DecayPop | 0.042325 | 0.035241–0.049781 | 4124 |
| SASRec − DecayPop | 0.012137 | 0.008292–0.016416 | 4124 |

上位3種はテスト順位からの事後的な抜粋。全モデル対も保存し、区間は多重比較未補正。区間が0を含む場合は優劣を確定しない。

## 実行状況

| モデル | 状況 |
|---|---|
| MostPop | completed |
| DecayPop | completed |
| ACT-R | completed |
| AudioCosine | completed |
| ItemKNN | completed |
| iALS | completed |
| BPR | completed |
| SASRec | completed |
| SANSA | not_run_resource_limit |

SANSAは公式50M listensの要件が100GB以上のRAMで、本機は24GB。149日制限版の必要メモリは未計測。曲数・密度を勝手に減らした結果は載せない。

`execution_status.json` の実行リストは今回の再開分だけ。ItemKNNは以前に29日/149日の学習と全96行の評価を完了しており、`itemknn_model_metrics.csv` と `04_itemknn.ipynb` に保存済み。

## 補助分析の出力

- `all_model_results.csv`：L全4値、T1/T2、B/C、検証/テスト、生スコア（主）とヘッド2種（副）。
- `model_head_coefficients.csv`：全モデル・全シード・全L・B/C・T1/T2の標準化後係数と符号。
- [SCORE_PROTOCOL_REPORT.md](SCORE_PROTOCOL_REPORT.md)：生スコアとヘッドの全モデル比較、7日再生数、学習CDFパーセンタイルの補助表。
- `task_a_next_item.csv`、`model_rank_comparison.csv`、`kendall_rank_agreement.csv`：次曲予測の順位とB/C順位。
- `rq3_summary.csv`（シード平均）、`rq3_metrics.csv`（各シード）：活動量、人気度、履歴長、好みへの近さ、初回再生率、24時間内Likeによる層別。
- `model_cold_coverage.csv`：各モデルの学習フィルタ後に最低スコアとなったペアの割合。
- `actr_base_metrics.csv`：ACT-Rの自発のみ・推薦のみbase-level（主9種とは別表）。
- `actr_component_metrics.csv`、`actr_component_diagnostics.csv`、`actr_component_paired_vs_audio.csv`：T2のSのみ・Pのみ・organic入力版とAudioCosineとの差。
- [ACTR_COMPONENT_REPORT.md](ACTR_COMPONENT_REPORT.md)：成分の実測、ヘッドの符号、欠損文脈と候補数の診断。
- `paired_model_differences.csv`、`top_model_paired_differences.csv`：全モデル対と上位3種の対応付きユーザー内AUC差。
- `sasrec_end_to_end_metrics.csv`：直接ラベル学習の補助実験。ファイルがなければ未実行。

## 解釈上の制約

- 表現モデルは学習ペアに29日目、評価ペアに149日目を使う。起点後の履歴は特徴に含めない。表現の学習量に分布差がある。
- 公式NDCGの分母のバグを標準DCG/IDCGへ補正。公式とビット単位で同じ調整結果とは主張しない。
- AはYambda型の複数正解の時系列holdout（150〜210日）。逐次更新するオンライン次曲予測の精度ではない。
- RQ3の将来活動と24時間内Likeは記述的な層別だけに使う。予測入力ではない。
- 再聴は観測された行動であり、満足度や推薦の因果効果そのものとは解釈しない。
