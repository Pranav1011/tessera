"""Phase 2: resumable, deduping, sharded ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh

from tessera.dataset import make_mesh
from tessera.ingest import ingest


def _make_files(d: Path, n: int = 8) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    kinds = ["plane", "sphere", "cylinder", "cone", "torus"]
    paths = []
    for i in range(n):
        m = make_mesh(kinds[i % len(kinds)], rng)
        p = d / f"m{i}.obj"
        m.export(p)
        paths.append(str(p))
    return paths


def _row_count(out: Path) -> int:
    return sum(pq.read_table(s).num_rows for s in out.glob("shard_*.parquet"))


def test_ingest_writes_parquet_records(tmp_path):
    paths = _make_files(tmp_path / "in")
    out = tmp_path / "out"
    r = ingest(paths, out, workers=2, shard_size=100, sample=False)
    assert r["total_processed"] == len(paths)
    assert list(out.glob("shard_*.parquet"))
    tbl = pq.read_table(next(iter(out.glob("shard_*.parquet"))))
    assert "is_valid" in tbl.column_names and tbl.num_rows > 0


def test_ingest_is_resumable(tmp_path):
    paths = _make_files(tmp_path / "in", n=8)
    out = tmp_path / "out"
    ingest(paths[:4], out, workers=2, sample=False)      # interrupted after 4
    r = ingest(paths, out, workers=2, sample=False)       # resume with the full list
    assert r["processed_this_run"] == 4                   # only the remaining 4 ran
    assert r["total_processed"] == 8


def test_ingest_dedupes_identical_geometry(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    m = trimesh.creation.icosphere(subdivisions=2)
    m.export(d / "a.obj")
    m.export(d / "b.obj")
    out = tmp_path / "out"
    ingest([str(d / "a.obj"), str(d / "b.obj")], out, workers=1, sample=False)
    assert _row_count(out) == 1  # byte-identical geometry deduped
