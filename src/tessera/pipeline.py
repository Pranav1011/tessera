"""Per-file driver: take a mesh path in, emit a typed record out.

load -> content-hash -> validate. Invalid models are quarantined with their typed
error list (never sampled). Valid models are canonicalized and sampled; the sampled
tensors are written next to the record so downstream training reads ML-ready shards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tessera.canonicalize import canonicalize
from tessera.sample import sample_sdf, sample_surface
from tessera.schema import GeometryRecord
from tessera.validate import genus_of, validate


def content_hash(mesh: trimesh.Trimesh) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices, np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, np.int64).tobytes())
    return h.hexdigest()[:16]


def process_mesh(mesh: trimesh.Trimesh, source_path: str) -> GeometryRecord:
    errors = [str(e) for e in validate(mesh)]
    return GeometryRecord(
        source_path=source_path,
        content_hash=content_hash(mesh),
        n_vertices=len(mesh.vertices),
        n_faces=len(mesh.faces),
        bbox_extent=float(mesh.bounding_box.extents.max()),
        is_valid=len(errors) == 0,
        errors=errors,  # type: ignore[arg-type]
        genus=genus_of(mesh),
    )


def process_file(path: str, sample_dir: str | None = None, n_points: int = 2048) -> dict[str, Any]:
    """Full per-file processing. Returns the record dict; writes sampled tensors for
    valid meshes to `sample_dir` if given."""
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
        # Weld coincident vertices: OBJ tessellations (e.g. ABC) store per-face
        # vertices, so raw topology looks non-manifold/non-watertight until merged.
        # This is standard import, not defect repair — structural checks run after.
        mesh.merge_vertices()
    except Exception as e:
        return {"source_path": path, "content_hash": "", "is_valid": False,
                "errors": [f"LOAD_ERROR:{type(e).__name__}"], "n_vertices": 0,
                "n_faces": 0, "bbox_extent": 0.0, "genus": None}
    rec = process_mesh(mesh, path)
    if rec.is_valid and sample_dir:
        cm = canonicalize(mesh)
        pts, nrm = sample_surface(cm, n_points)
        out = Path(sample_dir) / f"{rec.content_hash}.npz"
        payload = {"points": pts, "normals": nrm}
        sdf = sample_sdf(cm, n_points)
        if sdf is not None:
            payload["sdf_points"], payload["sdf"] = sdf
        np.savez_compressed(out, **payload)
    return rec.to_dict()
