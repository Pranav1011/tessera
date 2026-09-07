# Engineering Log — Tessera

## 2026-09-07 — MVP scope: pipeline, validation, eval framework; no GPU
**Goal:** a CPU-only geometry pipeline (ingestion, validation, slice-based eval).
Deep point-cloud models (PointNet, CUDA/DDP training) are out of scope: they need a
GPU and aren't required to exercise the data and eval infrastructure.

**Decisions:**
- **Mesh-first, not B-rep-first.** ABC ships OBJ + STEP; OpenCASCADE/pythonOCC
  (STEP) is the classic install/API sink. The validator, features, and eval all run
  on meshes via `trimesh`, so STEP parsing becomes an optional add-on, not a blocker.
- **Self-intersection done right.** A naive AABB-overlap screen false-positives on
  any curved closed mesh (a sphere flags itself). Added a Möller-Trumbore
  triangle-triangle narrow-phase so only real crossings fire — verified the clean
  icosphere passes and two piercing triangles are caught.
- **CPU model on purpose.** The eval *framework* is the deliverable, not accuracy, so
  the model is gradient-boosted trees on per-face geometry features (curvature via
  dihedral stats + neighbour-normal covariance eigenvalues) — CPU, minutes.
- **Grouped splits.** Train/test split by mesh, never by face, so no face leaks.
- **Honesty about synthetic numbers.** Synthetic primitives are trivially separable
  (0.93-1.0 mIoU); documented as a stand-in. Real numbers come from ABC.

**Built + tested (15 tests):** validator + taxonomy, features, synthetic dataset,
XGBoost baseline, slice eval with 3-seed CIs + regression gate.

## 2026-09-07 (cont.) — Phase 1b + 2: pipeline code-complete (24 tests)
Added canonicalization, surface/SDF sampling, content-hash, per-file driver, and
the distributed resumable ingestion → Parquet. Two bugs caught by tests, not review:
- **macOS `spawn` re-imports `__main__`** → workers died trying to run `<stdin>`/pytest.
  Fixed with a `fork` mp-context so workers inherit imports. (BrokenProcessPool → gone.)
- `GeometryRecord.to_dict()` called `.value` on errors, but the pipeline stores them as
  strings — only surfaced through the ingest path (the unit tests used `.errors`
  directly). Made it accept both enum and str.
**Demonstrated:** serial 27.5K → 6-worker 126K models/hr = **4.58x** (synthetic, sampling on).
**Status:** pipeline, validation and eval are built and tested; next is a run on
real ABC data.

## 2026-09-07 (cont.) — real ABC run caught a validator bug (the whole thesis, live)
Downloaded ABC chunk 0000 OBJ (7.9GB), extracted a 300-model sample, deleted the
archive (34GB disk can't hold full chunks). **First pass quarantined 99.7%** — a
BUG, not reality: ABC OBJs are un-welded per-face tessellations, so raw topology
reads non-manifold / non-watertight for nearly every model. Synthetic tests missed
it (trimesh primitives are pre-welded). Fixes: (1) `merge_vertices()` on import
(standard welding, not defect repair); (2) self-intersection face cap 100K→10K (it
was the ~47-min bottleneck). Result: **1.2s/model (8× faster)**.

**Real numbers (290 models after dedupe, chunk 0000):** 25.2% quarantined
[NOT_WATERTIGHT 47, NON_MANIFOLD 45, GENUS_ANOMALY 22, INCONSISTENT_NORMALS 3,
UNIT_SCALE 3, DEGENERATE 2], 0 load crashes, median 39K / max 2.27M faces,
throughput serial 5.5K → 6-worker 17.3K models/hr = 3.14×. Sample = 290 (disk-limited),
not full chunks.

**Takeaway:** synthetic tests passed while real data failed at 99.7%. Real data
belongs in the test loop as early as possible.
