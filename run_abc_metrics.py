"""Compute real bullet-1/bullet-2 metrics on the extracted ABC sample.

Validation stats (quarantine %, error taxonomy, crash rate) over all sampled
models; throughput serial-vs-parallel on a subset (serial over all of them would
take many minutes on large CAD meshes). Writes results/abc_metrics.json.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from tessera.ingest import ingest

ROOT = Path(__file__).resolve().parent


def main() -> None:
    paths = sorted(glob.glob(str(ROOT / "data/abc_sample/**/*.obj"), recursive=True))
    print(f"real ABC models: {len(paths)}", flush=True)

    # full validation pass (parallel)
    r_all = ingest(paths, ROOT / "data/out_all", workers=6, sample=False, resume=False)
    print(f"validation pass done: {r_all['seconds']}s", flush=True)

    # speedup on a subset (fair serial-vs-parallel comparison)
    sub = paths[:60]
    r1 = ingest(sub, ROOT / "data/out_s1", workers=1, sample=False, resume=False)
    r6 = ingest(sub, ROOT / "data/out_s6", workers=6, sample=False, resume=False)

    rows = []
    for s in glob.glob(str(ROOT / "data/out_all/shard_*.parquet")):
        rows += pq.read_table(s).to_pylist()
    n = len(rows)
    valid = sum(r["is_valid"] for r in rows)
    errc = Counter(e for r in rows for e in r["errors"])
    crashes = sum(1 for r in rows if any(e.startswith("LOAD_ERROR") for e in r["errors"]))
    faces = sorted(r["n_faces"] for r in rows if r["n_faces"])

    out = {
        "source": "ABC dataset chunk 0000, first 300 models (real CAD geometry)",
        "models": n,
        "valid": valid,
        "quarantined": n - valid,
        "quarantined_pct": round(100 * (n - valid) / n, 1) if n else 0,
        "load_crashes": crashes,
        "error_breakdown": dict(errc.most_common()),
        "median_faces": faces[len(faces) // 2] if faces else 0,
        "max_faces": faces[-1] if faces else 0,
        "throughput_serial_per_hr": r1["models_per_hr"],
        "throughput_6worker_per_hr": r6["models_per_hr"],
        "speedup": round(r6["models_per_hr"] / max(r1["models_per_hr"], 1), 2),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "abc_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
