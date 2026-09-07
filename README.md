# Tessera — CAD Geometry Validation & Slice-Based Evaluation

A CPU-only pipeline for validating CAD geometry, ingesting it at scale, and
**slice-based evaluation** of a primitive-segmentation model. Built mesh-first
(trimesh): no GPU, and no OpenCASCADE required for the core.

## Results on real CAD data (ABC dataset)

Run on a 290-model sample of [ABC](https://deep-geometry.github.io/abc-dataset/)
chunk 0000 (300 models, deduplicated), with `python run_abc_metrics.py` →
[`results/abc_metrics.json`](results/abc_metrics.json):

| Metric | Result |
|---|---|
| Quarantined by the typed validator | **73 / 290 (25.2%)** |
| Load crashes | 0 |
| Error breakdown | NOT_WATERTIGHT 47 · NON_MANIFOLD_EDGE 45 · GENUS_ANOMALY 22 · INCONSISTENT_NORMALS 3 · UNIT_SCALE_IMPLAUSIBLE 3 · DEGENERATE_FACE 2 |
| Mesh size | median 39,376 faces, max 2,271,312 |
| Ingestion throughput | 5,504 → **17,263 models/hr** with 6 workers (**3.14×**) |

**What broke first:** the first real run quarantined 99.7% of models. That was a bug,
not the data: ABC OBJs are un-welded per-face tessellations, so raw topology reads
as non-manifold for nearly everything. Synthetic tests missed it because trimesh
primitives come pre-welded. Welding vertices on import fixed it, and capping the
self-intersection screen made validation 8× faster (1.2 s/model). Details are in
[`ENGINEERING_LOG.md`](ENGINEERING_LOG.md).

## What's in it

| Component | Module | Tests |
|---|---|---|
| 9-rule typed geometry validator + error taxonomy (manifold, watertight, self-intersection, normals, unit/scale, genus, NaN, degenerate, duplicate) | `validate.py`, `schema.py` | `test_validate.py` |
| Canonicalization (centre / PCA / unit sphere), surface + SDF sampling, content hashing | `canonicalize.py`, `sample.py`, `pipeline.py` | `test_pipeline.py` |
| Parallel, resumable, deduplicating ingestion → Parquet shards (`ProcessPoolExecutor`) | `ingest.py` | `test_ingest.py` |
| Per-face geometric features + CPU XGBoost primitive-segmentation baseline | `features.py`, `model.py`, `dataset.py` | — |
| Slice-based eval: per-class IoU, slices by face count / genus / primitive, 3-seed CIs, CI regression gate | `eval_harness.py` | `test_eval.py` |

Self-intersection uses a real triangle-triangle test (Möller-Trumbore), not an AABB
screen that would flag every curved closed mesh.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
PYTHONPATH=src pytest -q            # 24 tests
PYTHONPATH=src python run_eval.py   # multi-seed slice eval -> results/eval.json

# Real data: download an ABC chunk (see scripts/download_abc.py), extract, then
python run_abc_metrics.py           # -> results/abc_metrics.json
```

## Limitations

- **Sample size:** the ABC numbers come from a 290-model sample of one chunk
  (disk-limited), not full chunks.
- **Self-intersection on large meshes:** the narrow-phase check runs only on meshes
  up to 10K faces (`validate.py`), so most ABC models (median 39K faces) skip it.
- **Segmentation accuracy is synthetic only:** the XGBoost baseline scores 0.93 mIoU on
  synthetic primitives, which are trivially separable. The deliverable is the eval
  framework, not state-of-the-art accuracy, and there is no mIoU on ABC yet.
