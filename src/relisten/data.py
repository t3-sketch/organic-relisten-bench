"""50M の取得・検証。全曲埋め込みのダウンロードは行わない。"""
import hashlib
import json
from pathlib import Path

import duckdb
import httpx
import polars as pl

from .config import RAW, PROCESSED, RESULTS, SECONDS_PER_DAY


def connection():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'")
    con.execute("SET threads=4")
    con.execute(f"SET temp_directory='{PROCESSED / 'tmp'}'")
    return con


def ensure_raw():
    """リビジョンと SHA256 を固定・記録し、公開 50M だけを取得する。"""
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.get("https://huggingface.co/api/datasets/yandex/yambda?blobs=true")
            response.raise_for_status()
            info = response.json()
        manifest = {"repo": "yandex/yambda", "revision": info["sha"], "files": {}}
        for name in ("flat/50m/listens.parquet", "flat/50m/likes.parquet"):
            item = next(s for s in info["siblings"] if s["rfilename"] == name)
            manifest["files"][name] = {"size": item["size"], "sha256": item["lfs"]["sha256"]}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, expected in manifest["files"].items():
        path = RAW / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://huggingface.co/datasets/yandex/yambda/resolve/{manifest['revision']}/{name}"
            partial = path.with_suffix(".parquet.partial")
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        output.write(chunk)
            partial.replace(path)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if path.stat().st_size != expected["size"] or digest != expected["sha256"]:
            raise ValueError(f"公開データとの照合に失敗: {path}")
    return manifest


def prepare_data():
    manifest = ensure_raw()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    listens = pl.scan_parquet(RAW / "flat/50m/listens.parquet")
    required = {"uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"}
    if not required.issubset(listens.collect_schema().names()):
        raise ValueError("必要な再生列が不足")
    summary = listens.select(
        pl.len().alias("listens"), pl.col("uid").n_unique().alias("users"),
        pl.col("item_id").n_unique().alias("items"),
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
        pl.col("is_organic").mean().alias("organic_fraction"),
        pl.col("is_organic").is_in([0, 1]).all().alias("valid_routes"),
        pl.any_horizontal(pl.all().is_null()).any().alias("has_nulls"),
    ).collect()
    if summary["has_nulls"][0] or not summary["valid_routes"][0]:
        raise ValueError("再生データに欠損または未知の経路")
    summary.write_csv(RESULTS / "dataset_summary.csv")
    listens.select("item_id").unique().sink_parquet(PROCESSED / "items_50m.parquet")
    listens.with_columns((pl.col("timestamp").cast(pl.Int64) // SECONDS_PER_DAY).alias("day")).group_by(
        "uid", "day"
    ).agg(
        pl.col("timestamp").min().alias("first_timestamp"),
        pl.col("timestamp").max().alias("last_timestamp"),
        pl.len().alias("listens"),
    ).sort("uid", "day").sink_parquet(PROCESSED / "user_days.parquet")
    # データ自体を将来の学習で読み違えないよう、カットオフを物理的にも分ける。
    from .splits import representation_history
    representation_history(listens).sink_parquet(PROCESSED / "representation_train.parquet")
    return summary, manifest


def embedding_coverage(revision):
    """HTTP range + Parquet 列選択で item_id だけを読む（ベクトル列は取得しない）。"""
    path = PROCESSED / "embedding_item_ids_50m.parquet"
    with connection() as con:
        if not path.exists():
            con.execute("INSTALL httpfs; LOAD httpfs")
            url = f"https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/embeddings.parquet"
            con.execute(
                f"COPY (SELECT DISTINCT e.item_id FROM read_parquet(?) e "
                f"SEMI JOIN read_parquet(?) i USING (item_id)) TO '{path}' (FORMAT PARQUET)",
                [url, str(PROCESSED / "items_50m.parquet")],
            )
    n_items = pl.scan_parquet(PROCESSED / "items_50m.parquet").select(pl.len()).collect().item()
    n_embedded = pl.scan_parquet(path).select(pl.len()).collect().item()
    table = pl.DataFrame({"items": [n_items], "embedded_items": [n_embedded],
                          "fraction": [n_embedded / n_items], "revision": [revision]})
    table.write_csv(RESULTS / "embedding_coverage.csv")
    return table


def prepare_audio_embeddings():
    """約8GBの正規化列だけを読み、50M分をfloat32で保存する。"""
    path = PROCESSED / "audio_embeddings_50m.parquet"
    if path.exists():
        return path
    revision = json.loads((RAW / "manifest.json").read_text())["revision"]
    url = f"https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/embeddings.parquet"
    partial = path.with_suffix(".parquet.partial")
    with connection() as con:
        con.execute("LOAD httpfs")
        con.execute(
            f"COPY (SELECT e.item_id, e.normalized_embed::FLOAT[] AS normalized_embed "
            f"FROM read_parquet(?) e SEMI JOIN read_parquet(?) i USING(item_id)) "
            f"TO '{partial}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            [url, str(PROCESSED / "items_50m.parquet")],
        )
    partial.replace(path)
    return path
