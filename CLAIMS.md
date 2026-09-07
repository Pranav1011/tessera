# CLAIMS — Tessera
Updated 2026-09-07. Synthetic-data numbers are stand-ins; real numbers come from ABC.

| Claim | Status | Evidence artifact | Reproduce | Verified |
|---|---|---|---|---|
| 9-rule typed geometry validator (manifold, watertight, self-intersection, normals, unit/scale, genus, NaN, degenerate, duplicate) | VERIFIED | `src/tessera/validate.py`, `src/tessera/schema.py`, `tests/test_validate.py` | `pytest tests/test_validate.py` | 2026-09-07 |
| Self-intersection is a *real* triangle-triangle test (Möller-Trumbore), not an AABB false-positive | VERIFIED | `_tris_intersect` + `test_self_intersection_detected` + `test_clean_mesh_passes_all_rules` | `pytest -k intersection` | 2026-09-07 |
| Per-face features + CPU XGBoost primitive-segmentation baseline (no GPU) | VERIFIED | `src/tessera/features.py`, `model.py`, `dataset.py` | `python run_eval.py` | 2026-09-07 |
| Slice-based eval: per-class IoU + slices (face-count/genus/primitive) + 3-seed CIs + CI regression gate | VERIFIED | `src/tessera/eval_harness.py`, `tests/test_eval.py`, `run_eval.py`, `results/eval.json` | `pytest tests/test_eval.py; python run_eval.py` | 2026-09-07 |
| Canonicalization (center/PCA/unit-sphere) + surface & SDF sampling + content-hash + per-file driver | VERIFIED | `canonicalize.py`, `sample.py`, `pipeline.py`, `tests/test_pipeline.py` | `pytest tests/test_pipeline.py` | 2026-09-07 |
| **Distributed, resumable, deduping ingestion → Parquet shards; 4.58x parallel speedup** (synthetic) | VERIFIED | `src/tessera/ingest.py`, `tests/test_ingest.py` | `pytest tests/test_ingest.py` | 2026-09-07 |
| 24 tests pass, ruff clean | VERIFIED | full suite | `pytest -q; ruff check src tests` | 2026-09-07 |
| **REAL ABC metrics** (290-model sample, chunk 0000): **25.2% quarantined** under the typed taxonomy (NOT_WATERTIGHT 47, NON_MANIFOLD 45, GENUS_ANOMALY 22, …), 0 load crashes, median 39K / max 2.27M faces; ingestion **3.14× speedup (5.5K→17.3K models/hr)** | VERIFIED | `results/abc_metrics.json`, `run_abc_metrics.py` | download an ABC chunk, extract, `python run_abc_metrics.py` | 2026-09-07 |

**Not claimable:** synthetic-data numbers (mIoU ~0.93, 4.58× speedup, 126K models/hr) are stand-ins on trivially separable data. Only the ABC row above is a real-data result. Still open: sliced mIoU and named failure modes on ABC.
