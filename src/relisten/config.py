from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LOWER_BOUND_DAYS = 1  # TM がモデル結果を見る前に概念上の理由で決定。下端を含む。
HORIZON_DAYS = 90
SECONDS_PER_DAY = 86400  # 公式の分割コードは生 timestamp を秒として扱う。
REPRESENTATION_END_DAY = 149
HEAD_REPRESENTATION_END_DAY = 29  # TM承認：30〜59日の学習用スコアに未来を混ぜない。
SPLITS = {"train": (30, 59), "validation": (150, 164), "test": (165, 210)}
SENSITIVITY_LOWER_DAYS = (1, 3, 7, 14)
SEED = 20261016
SOURCE_URL = "https://app.notion.com/p/3f680c3dba2381509bdfd6ea0ff64d5b"
BENCHMARK_MODELS = ("MostPop","DecayPop","ItemKNN","iALS","BPR","SANSA","SASRec","ACT-R","AudioCosine")
