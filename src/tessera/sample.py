"""Representation sampling: surface point cloud + normals (always), and a
signed-distance / occupancy query set (only when the mesh is watertight, since sign
is undefined otherwise). These are the ML-ready tensors the pipeline shards out."""

from __future__ import annotations

import numpy as np
import trimesh


def sample_surface(mesh: trimesh.Trimesh, n: int = 2048, seed: int = 0):
    """Return (points [n,3], normals [n,3]) sampled on the surface."""
    rng = np.random.default_rng(seed)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    normals = np.asarray(mesh.face_normals)[face_idx]
    return np.asarray(pts, np.float32), np.asarray(normals, np.float32)


def sample_sdf(mesh: trimesh.Trimesh, n: int = 2048, seed: int = 0):
    """Return (query_points [n,3], sdf [n]) in the unit cube, or None if the mesh
    isn't watertight (sign undefined)."""
    if not mesh.is_watertight:
        return None
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    try:
        sdf = trimesh.proximity.signed_distance(mesh, q).astype(np.float32)
    except Exception:
        return None
    return q, sdf
