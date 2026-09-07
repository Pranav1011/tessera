"""Distributed, resumable ingestion (Phase 2).

A process worker pool runs the per-file pipeline; records stream into content-keyed
Parquet shards. Resumable: a checkpoint of processed source paths is written after
every shard, so `kill -9` mid-run and re-run picks up where it stopped. Content-hash
dedupe drops byte-identical geometry. (A `ProcessPoolExecutor` is used here; Ray is a
drop-in for multi-node scale — same `process_file` task.)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tessera.pipeline import process_file

# fork avoids spawn's re-import of __main__ (which breaks under pytest / -c / stdin
# on macOS, where spawn is the default). Workers just inherit the parent's imports.
_MP_CTX = mp.get_context("fork") if sys.platform != "win32" else None

_CKPT = "_processed.json"


def _load_done(out: Path) -> set[str]:
    p = out / _CKPT
    return set(json.loads(p.read_text())) if p.exists() else set()


def _save_done(out: Path, done: list[str]) -> None:
    (out / _CKPT).write_text(json.dumps(done))


def _write_shard(out: Path, records: list[dict], shard_idx: int) -> None:
    pq.write_table(pa.Table.from_pylist(records), out / f"shard_{shard_idx:05d}.parquet")


def ingest(paths, out_dir, workers: int = 4, shard_size: int = 500,
           sample: bool = True, resume: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_dir = None
    if sample:
        sample_dir = str(out / "samples")
        Path(sample_dir).mkdir(exist_ok=True)

    done = _load_done(out) if resume else set()
    todo = [p for p in paths if p not in done]
    processed = list(done)
    seen_hashes: set[str] = set()
    shard_idx = len(list(out.glob("shard_*.parquet")))
    buffer: list[dict] = []
    t0 = time.time()

    fn = partial(process_file, sample_dir=sample_dir)
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CTX) as ex:
        for rec in ex.map(fn, todo):
            processed.append(rec["source_path"])
            h = rec.get("content_hash")
            if h and h in seen_hashes:
                continue  # content-hash dedupe
            if h:
                seen_hashes.add(h)
            buffer.append(rec)
            if len(buffer) >= shard_size:
                _write_shard(out, buffer, shard_idx)
                shard_idx += 1
                buffer = []
                _save_done(out, processed)
    if buffer:
        _write_shard(out, buffer, shard_idx)
    _save_done(out, processed)

    elapsed = time.time() - t0
    return {
        "processed_this_run": len(todo),
        "total_processed": len(processed),
        "seconds": round(elapsed, 3),
        "models_per_hr": round(len(todo) / elapsed * 3600) if elapsed > 0 else 0,
        "shards": shard_idx + (1 if False else 0),
    }
