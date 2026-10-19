# Paper 1 実験結果

主設定：L=1、推薦再生から24時間以上〜90日後（両端を含む）の自発再聴。
TMがモデル結果を見る前に概念上の理由で決定。L=7・14は必須感度分析として保持。

## T2：未知曲の推薦後の自発再聴

スコア単独を共通ロジスティック回帰に入力した主結果。ユーザーを等重みとし、正負両方を持つユーザーのみAUCを計算。

| モデル | ユーザー内AUC | 95%CI | 全体AUC | PR-AUC |
|---|---:|---|---:|---:|
| AudioCosine | 0.5619 | 0.5570–0.5668 | 0.5962 | 0.1291 |
| SASRec | 0.5317 | 0.5269–0.5366 | 0.5196 | 0.1062 |
| DecayPop | 0.5200 | 0.5147–0.5249 | 0.5262 | 0.1224 |
| ACT-R | 0.5105 | 0.5064–0.5150 | 0.5659 | 0.1255 |
| MostPop | 0.5074 | 0.5022–0.5124 | 0.5192 | 0.1151 |
| BPR | 0.5071 | 0.5027–0.5116 | 0.5380 | 0.1090 |
| ItemKNN | 0.5005 | 0.4957–0.5050 | 0.4714 | 0.0979 |
| iALS | 0.4989 | 0.4945–0.5036 | 0.5396 | 0.1092 |

評価対象は967,884ペア、4,819ユーザー。AUCを計算できるのは4,124ユーザー。ユーザー基礎率だけのユーザー内AUCは0.5。

iALS/BPRは3シードの平均。信頼区間は各ユーザーのシード平均AUCを1000回のユーザーブートストラップで求める。
モデル間の対応付き差は `paired_model_differences.csv`。区間が0を含む差を優劣確定と扱わない。

## 下限の感度分析

| モデル | L=1 | L=3 | L=7 | L=14 |
|---|---:|---:|---:|---:|
| AudioCosine | 0.5619 | 0.5636 | 0.5649 | 0.5643 |
| SASRec | 0.5317 | 0.5321 | 0.5331 | 0.5353 |
| DecayPop | 0.5200 | 0.5201 | 0.5201 | 0.5224 |
| ACT-R | 0.5105 | 0.5095 | 0.5094 | 0.5106 |
| MostPop | 0.5074 | 0.5075 | 0.5075 | 0.5098 |
| BPR | 0.5071 | 0.5075 | 0.5074 | 0.5055 |
| ItemKNN | 0.5005 | 0.5005 | 0.5007 | 0.5040 |
| iALS | 0.4989 | 0.4992 | 0.4991 | 0.4970 |

各Lで同じ窓の活動30日以上条件を再計算し、対応する学習ラベルで共通ヘッドを再学習。Lは結果を見て変更しない。

![下限の感度分析](figures/model_lower_bound_sensitivity.png)

## 上位モデル間の対応付き差

L=1・T2・C・スコア単独。各ユーザーのAUCの差を1000回、ユーザー単位で再標本化。iALS/BPRはユーザー内で先に3シードを平均する。

| 比較（左−右） | AUC差 | 95%CI | ユーザー数 |
|---|---:|---|---:|
| AudioCosine − SASRec | 0.030188 | 0.023298–0.037633 | 4124 |
| AudioCosine − DecayPop | 0.041899 | 0.034798–0.049346 | 4124 |
| SASRec − DecayPop | 0.011711 | 0.007878–0.016022 | 4124 |

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

- `all_model_results.csv`：L全4値、T1/T2、B/C、検証/テスト、スコア単独とユーザー基礎率追加。
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
